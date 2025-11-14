import argparse, time, sys, os
from collections import deque
import eventlet; eventlet.monkey_patch()
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# 프로젝트 루트를 Python 경로에 추가
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from audio.stream import AudioStream
from audio.vad import VADGate
from tempo.beat import BeatEstimator
from align.matcher import LineMatcher, tail_token_coverage
from structure.chorus import detect_sections
from profiles.profile_store import ProfileStore
from stt.engine_whisper import WhisperEngine
from stt.engine_vosk import VoskEngine

app = Flask(__name__, template_folder='templates', static_folder='static')
socketio = SocketIO(app, cors_allowed_origins='*')
FORCE_GO=False; HOLD=False

def load_lyrics(path='lyrics.txt'):
    with open(path,'r',encoding='utf-8') as f:
        lines=[ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]

@app.route('/')
def index(): return render_template('index.html')

@socketio.on('cmd')
def on_cmd(data):
    global FORCE_GO, HOLD
    t=(data or {}).get('type')
    if t=='GO': FORCE_GO=True; emit('ack',{'ok':True,'type':t})
    elif t=='HOLD_TOGGLE': HOLD=not HOLD; emit('ack',{'ok':True,'type':t,'hold':HOLD})
    else: emit('ack',{'ok':False,'type':t})

def run_loop(args):
    global FORCE_GO, HOLD
    lyrics=load_lyrics(args.lyrics)
    if not lyrics: print('[err] lyrics.txt is empty.'); return

    store=ProfileStore('profiles'); text='\n'.join(lyrics); prof=store.load(text) or {}
    lead_ms=prof.get('lead_ms', args.lead_ms); th_lock=prof.get('th_lock', args.lock)
    th_preview=prof.get('th_preview', args.preview); th_release=prof.get('th_release', args.release)
    advance_score=prof.get('advance_score', args.advance_score); advance_cover=prof.get('advance_cover', args.advance_cover)

    mic=AudioStream(samplerate=args.sr, block_duration=args.chunk, channels=2 if args.input=='system' else 1,
                    device_name=args.device_name, device_index=args.device_index, native_rate=args.native_rate)
    vad=VADGate(samplerate=args.sr, aggressiveness=args.vad_level) if args.vad else None
    beat=BeatEstimator(samplerate=args.sr, win_seconds=max(4.0, args.history))
    matcher=LineMatcher(th_lock=th_lock, th_preview=th_preview, th_release=th_release)
    labels, blocks = detect_sections(lyrics)
    recent=deque(maxlen=int(args.history/args.chunk)); line_idx=0; locked_idx=-1

    if args.engine=='whisper':
        stt=WhisperEngine(model_size=args.model, device=args.device, compute_type='float16' if args.device=='metal' else 'int8'); vosk=None
    else:
        stt=None; vosk=VoskEngine(model_dir=args.vosk_model, samplerate=args.sr)
        if args.strict_order: vosk.update_grammar(lyrics[line_idx], '')
        else: vosk.update_grammar(lyrics[line_idx], lyrics[line_idx+1] if line_idx+1<len(lyrics) else '')

    mic.start(); time.sleep(0.2)
    try:
        while line_idx < len(lyrics):
            pcm=mic.read(); beat.update(pcm)
            if vad and not vad.is_speech(pcm):
                socketio.emit('state',{
                    'current': lyrics[line_idx],
                    'next': lyrics[line_idx+1] if (line_idx+1<len(lyrics)) else '(끝)',
                    'locked': False, 'progress':0, 'score_curr':0,'score_next':0, 'partial':'(무성 구간)',
                    'label': labels[line_idx],
                    'section': 'Chorus' if labels[line_idx].startswith('C') else ('Verse' if labels[line_idx].startswith('V') else ('Bridge' if labels[line_idx].startswith('B') else 'Other')),
                    'lead_ms': lead_ms
                }); continue
            text = stt.transcribe_chunk(pcm, samplerate=args.sr) if stt else vosk.accept_audio(pcm)
            if text: recent.append(text)
            recent_text=' '.join(list(recent)[-3:])
            locked_idx, show_preview = matcher.decide(recent_text, lyrics, line_idx)
            if not show_preview and beat.early_display_hint(lead_ms=lead_ms): show_preview=True

            score_curr = matcher.score(recent_text, lyrics[line_idx])
            score_next = matcher.score(recent_text, lyrics[line_idx+1] if line_idx+1<len(lyrics) else '')
            progress = min(100.0, max(0.0, score_curr))
            socketio.emit('state',{
                'current': lyrics[line_idx],
                'next': lyrics[line_idx+1] if (show_preview and line_idx+1<len(lyrics)) else '(준비중...)',
                'locked': (locked_idx==line_idx),
                'progress': progress, 'score_curr':score_curr, 'score_next':score_next,
                'partial': (text[:40] + '…') if len(text)>40 else text,
                'label': labels[line_idx],
                'section': 'Chorus' if labels[line_idx].startswith('C') else ('Verse' if labels[line_idx].startswith('V') else ('Bridge' if labels[line_idx].startswith('B') else 'Other')),
                'lead_ms': lead_ms
            })
            if not HOLD and (locked_idx==line_idx):
                cover=tail_token_coverage(recent_text, lyrics[line_idx], tail_ratio=0.5)
                should_go=(score_curr>=advance_score) or (cover>=advance_cover) or FORCE_GO
                if should_go:
                    FORCE_GO=False; line_idx+=1
                    if vosk and line_idx<len(lyrics):
                        if args.strict_order: vosk.update_grammar(lyrics[line_idx], '')
                        else: vosk.update_grammar(lyrics[line_idx], lyrics[line_idx+1] if line_idx+1<len(lyrics) else '')
    except KeyboardInterrupt:
        print('\n[info] stopped by user')
    finally:
        mic.stop()
        out={'lead_ms':lead_ms,'th_lock':matcher.th_lock,'th_preview':matcher.th_preview,'th_release':matcher.th_release,
             'advance_score':advance_score,'advance_cover':advance_cover}
        store.save(text,out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--engine', default='whisper', choices=['whisper','vosk'])
    ap.add_argument('--model', default='small'); ap.add_argument('--device', default='metal')
    ap.add_argument('--vosk_model', default='models/ko'); ap.add_argument('--lyrics', default='lyrics.txt')
    ap.add_argument('--sr', type=int, default=16000); ap.add_argument('--chunk', type=float, default=1.0)
    ap.add_argument('--history', type=int, default=6)
    ap.add_argument('--lock', type=int, default=75); ap.add_argument('--preview', type=int, default=55); ap.add_argument('--release', type=int, default=50)
    ap.add_argument('--advance_score', type=int, default=88); ap.add_argument('--advance_cover', type=float, default=0.6)
    ap.add_argument('--vad', action='store_true'); ap.add_argument('--vad_level', type=int, default=2)
    ap.add_argument('--lead_ms', type=int, default=300); ap.add_argument('--strict_order', action='store_true')
    ap.add_argument('--input', default='mic', choices=['mic','system'])
    ap.add_argument('--device_name', default=None); ap.add_argument('--device_index', type=int, default=None); ap.add_argument('--native_rate', type=int, default=None)
    ap.add_argument('--host', default='0.0.0.0'); ap.add_argument('--port', type=int, default=5000)
    args=ap.parse_args()
    socketio.start_background_task(run_loop, args)
    print('[info] Web UI: http://localhost:%d' % args.port)
    socketio.run(app, host=args.host, port=args.port)
if __name__=='__main__': main()
