import json, os, hashlib, time
class ProfileStore:
    def __init__(self, dirpath='profiles'):
        self.dir=dirpath; os.makedirs(self.dir, exist_ok=True)
    def key_for_lyrics(self, lyrics_text:str):
        return hashlib.md5(lyrics_text.encode('utf-8')).hexdigest()
    def path(self, key:str): return os.path.join(self.dir, f'{key}.json')
    def load(self, lyrics_text:str):
        key=self.key_for_lyrics(lyrics_text); p=self.path(key)
        if os.path.exists(p):
            with open(p,'r',encoding='utf-8') as f: return json.load(f)
        return None
    def save(self, lyrics_text:str, data:dict):
        key=self.key_for_lyrics(lyrics_text); p=self.path(key)
        data={**data, '_updated': int(time.time())}
        with open(p,'w',encoding='utf-8') as f: json.dump(data,f,ensure_ascii=False,indent=2)
        return p
