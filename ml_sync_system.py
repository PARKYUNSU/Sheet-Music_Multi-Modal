"""
머신러닝 기반 악보-연주 싱크 시스템
기존 휴리스틱 매칭을 머신러닝 모델로 대체하여 정확도 향상
"""

import json
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from collections import deque
import librosa
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from improved_omr import ImprovedOMR
from real_time_audio_processor import RealTimeAudioProcessor

class MLSyncSystem:
    def __init__(self, sheet_music_path: str, audio_file_path: str):
        """
        머신러닝 기반 싱크 시스템 초기화
        
        Args:
            sheet_music_path: 악보 이미지 경로
            audio_file_path: 오디오 파일 경로
        """
        self.sheet_music_path = sheet_music_path
        self.audio_file_path = audio_file_path
        
        # OMR 시스템 초기화
        print("🎼 ML 기반 OMR 시스템 초기화...")
        self.omr = ImprovedOMR()
        self.sheet_music_data = None
        
        # 오디오 처리 시스템 초기화
        print("🎵 오디오 처리 시스템 초기화...")
        self.audio_processor = RealTimeAudioProcessor(audio_file_path)
        
        # 머신러닝 모델들
        self.models = {
            'sync_classifier': None,  # 동기화 여부 분류
            'position_regressor': None,  # 위치 예측 회귀
            'confidence_regressor': None,  # 신뢰도 예측 회귀
            'section_classifier': None  # 구간 분류
        }
        
        # 특성 스케일러
        self.scalers = {
            'audio_features': StandardScaler(),
            'sheet_features': StandardScaler(),
            'combined_features': StandardScaler()
        }
        
        # 훈련 데이터
        self.training_data = {
            'features': [],
            'labels': [],
            'positions': [],
            'confidences': [],
            'sections': []
        }
        
        # 동기화 상태
        self.sync_state = {
            'current_position': 0,
            'sync_confidence': 0.0,
            'is_synced': False,
            'current_section': 'unknown',
            'ml_predictions': deque(maxlen=10)  # 최근 10개 예측 결과
        }
        
        print("✅ ML 기반 악보-연주 싱크 시스템 초기화 완료")
    
    def extract_audio_features(self, audio_chunk: np.ndarray, sample_rate: int = 22050) -> Dict:
        """오디오 특성 추출"""
        features = {}
        
        # 1. 기본 음높이 특성
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_chunk, fmin=80, fmax=1000, sr=sample_rate
        )
        
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) > 0:
            features['pitch_mean'] = np.mean(valid_f0)
            features['pitch_std'] = np.std(valid_f0)
            features['pitch_min'] = np.min(valid_f0)
            features['pitch_max'] = np.max(valid_f0)
            features['pitch_range'] = features['pitch_max'] - features['pitch_min']
        else:
            features.update({
                'pitch_mean': 0, 'pitch_std': 0, 'pitch_min': 0,
                'pitch_max': 0, 'pitch_range': 0
            })
        
        # 2. 볼륨 특성
        rms = librosa.feature.rms(y=audio_chunk)[0]
        features['volume_mean'] = np.mean(rms)
        features['volume_std'] = np.std(rms)
        features['volume_max'] = np.max(rms)
        
        # 3. 스펙트럴 특성
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_chunk, sr=sample_rate)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # 4. MFCC 특성 (음색 특성)
        mfccs = librosa.feature.mfcc(y=audio_chunk, sr=sample_rate, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # 5. 리듬 특성
        onset_frames = librosa.onset.onset_detect(y=audio_chunk, sr=sample_rate)
        features['onset_count'] = len(onset_frames)
        if len(onset_frames) > 1:
            onset_intervals = np.diff(onset_frames)
            features['onset_interval_mean'] = np.mean(onset_intervals)
            features['onset_interval_std'] = np.std(onset_intervals)
        else:
            features['onset_interval_mean'] = 0
            features['onset_interval_std'] = 0
        
        # 6. 템포 특성
        tempo, beats = librosa.beat.beat_track(y=audio_chunk, sr=sample_rate)
        features['tempo'] = tempo
        features['beat_count'] = len(beats)
        
        return features
    
    def extract_sheet_features(self, note: Dict, context_notes: List[Dict]) -> Dict:
        """악보 특성 추출"""
        features = {}
        
        # 1. 음표 기본 특성
        note_freq = self.note_name_to_frequency(note['pitch'])
        features['note_frequency'] = note_freq
        features['note_duration'] = note.get('duration', 1.0)
        features['note_area'] = note.get('area', 0)
        features['note_aspect_ratio'] = note.get('aspect_ratio', 1.0)
        
        # 2. 음표 타입 특성 (원핫 인코딩)
        note_types = ['whole', 'half', 'quarter', 'eighth', 'sixteenth']
        for note_type in note_types:
            features[f'note_type_{note_type}'] = 1 if note.get('type') == note_type else 0
        
        # 3. 컨텍스트 특성 (주변 음표들)
        if len(context_notes) > 0:
            context_freqs = [self.note_name_to_frequency(n['pitch']) for n in context_notes]
            context_freqs = [f for f in context_freqs if f > 0]
            
            if context_freqs:
                features['context_pitch_mean'] = np.mean(context_freqs)
                features['context_pitch_std'] = np.std(context_freqs)
                features['context_pitch_range'] = np.max(context_freqs) - np.min(context_freqs)
            else:
                features.update({
                    'context_pitch_mean': 0, 'context_pitch_std': 0, 'context_pitch_range': 0
                })
            
            # 음정 관계
            if len(context_freqs) > 1:
                intervals = []
                for i in range(1, len(context_freqs)):
                    interval = 1200 * np.log2(context_freqs[i] / context_freqs[i-1])
                    intervals.append(interval)
                features['interval_mean'] = np.mean(intervals)
                features['interval_std'] = np.std(intervals)
            else:
                features['interval_mean'] = 0
                features['interval_std'] = 0
        
        # 4. 위치 특성
        features['note_position'] = note.get('index', 0)
        features['relative_position'] = note.get('index', 0) / max(1, len(context_notes))
        
        return features
    
    def create_training_data(self):
        """훈련 데이터 생성"""
        print("📊 훈련 데이터 생성 중...")
        
        # 악보 데이터 로드
        self.sheet_music_data = self.omr.process_sheet_music_improved(self.sheet_music_path)
        
        # 오디오 분석
        audio_analysis = self.audio_processor.analyze_full_audio()
        audio_results = audio_analysis['detailed_results']
        
        # 시간축 변환
        self.convert_notes_to_timeline()
        
        notes = self.sheet_music_data['notes']
        
        # 각 오디오 샘플에 대해 훈련 데이터 생성
        for i, audio_result in enumerate(audio_results):
            current_time = audio_result['time']
            current_pitch = audio_result['pitch']
            
            # 오디오 특성 추출
            # 실제로는 audio_chunk가 필요하지만, 여기서는 결과를 사용
            audio_features = {
                'pitch_mean': current_pitch,
                'pitch_std': 0,  # 단일 샘플이므로 0
                'volume_mean': audio_result['volume'],
                'volume_std': 0,
                'tempo': 120,  # 기본값
                'onset_count': 0,
                'spectral_centroid_mean': current_pitch * 2,  # 근사값
            }
            
            # 각 악보 음표와의 매칭 시도
            for j, note in enumerate(notes):
                # 악보 특성 추출
                context_notes = notes[max(0, j-2):min(len(notes), j+3)]
                sheet_features = self.extract_sheet_features(note, context_notes)
                
                # 결합된 특성
                combined_features = {**audio_features, **sheet_features}
                
                # 라벨 생성 (간단한 휴리스틱으로 시작)
                time_diff = abs(current_time - note['start_time'])
                pitch_diff = abs(current_pitch - sheet_features['note_frequency'])
                
                # 매칭 라벨 (임계값 기반)
                is_match = (time_diff < 5.0 and pitch_diff < 100)  # 5초, 100Hz 허용
                confidence = max(0, 1.0 - (time_diff / 5.0) - (pitch_diff / 100))
                
                # 구간 라벨 (시간 기반)
                if current_time < 10:
                    section = 'intro'
                elif current_time < 130:
                    section = 'verse1'
                else:
                    section = 'verse2'
                
                # 훈련 데이터에 추가
                self.training_data['features'].append(combined_features)
                self.training_data['labels'].append(1 if is_match else 0)
                self.training_data['positions'].append(j)
                self.training_data['confidences'].append(confidence)
                self.training_data['sections'].append(section)
        
        print(f"✅ 훈련 데이터 생성 완료: {len(self.training_data['features'])}개 샘플")
    
    def prepare_ml_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """머신러닝을 위한 특성 준비"""
        if not self.training_data['features']:
            raise ValueError("훈련 데이터가 없습니다. create_training_data()를 먼저 실행하세요.")
        
        # 특성을 벡터로 변환
        feature_names = list(self.training_data['features'][0].keys())
        X = np.array([[sample[feature] for feature in feature_names] 
                     for sample in self.training_data['features']])
        
        # 라벨들
        y_sync = np.array(self.training_data['labels'])
        y_positions = np.array(self.training_data['positions'])
        y_confidences = np.array(self.training_data['confidences'])
        y_sections = np.array(self.training_data['sections'])
        
        # 특성 스케일링
        X_scaled = self.scalers['combined_features'].fit_transform(X)
        
        return X_scaled, y_sync, y_positions, y_confidences, y_sections
    
    def train_models(self):
        """머신러닝 모델 훈련"""
        print("🤖 머신러닝 모델 훈련 시작...")
        
        # 특성 준비
        X, y_sync, y_positions, y_confidences, y_sections = self.prepare_ml_features()
        
        # 훈련/테스트 분할
        X_train, X_test, y_sync_train, y_sync_test = train_test_split(
            X, y_sync, test_size=0.2, random_state=42, stratify=y_sync
        )
        
        # 1. 동기화 분류기 훈련
        print("   - 동기화 분류기 훈련...")
        self.models['sync_classifier'] = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'
        )
        self.models['sync_classifier'].fit(X_train, y_sync_train)
        
        # 분류 성능 평가
        y_sync_pred = self.models['sync_classifier'].predict(X_test)
        sync_accuracy = accuracy_score(y_sync_test, y_sync_pred)
        print(f"     동기화 분류 정확도: {sync_accuracy:.3f}")
        
        # 2. 위치 회귀기 훈련 (동기화된 샘플만)
        print("   - 위치 회귀기 훈련...")
        sync_mask = y_sync == 1
        if np.sum(sync_mask) > 10:  # 충분한 양성 샘플이 있는 경우
            X_sync = X[sync_mask]
            y_pos_sync = y_positions[sync_mask]
            
            X_train_sync, X_test_sync, y_pos_train, y_pos_test = train_test_split(
                X_sync, y_pos_sync, test_size=0.2, random_state=42
            )
            
            self.models['position_regressor'] = RandomForestRegressor(
                n_estimators=100, random_state=42
            )
            self.models['position_regressor'].fit(X_train_sync, y_pos_train)
            
            y_pos_pred = self.models['position_regressor'].predict(X_test_sync)
            pos_mse = np.mean((y_pos_test - y_pos_pred) ** 2)
            print(f"     위치 회귀 MSE: {pos_mse:.3f}")
        
        # 3. 신뢰도 회귀기 훈련
        print("   - 신뢰도 회귀기 훈련...")
        # 신뢰도 데이터도 같은 분할 사용
        _, _, y_conf_train, y_conf_test = train_test_split(
            X, y_confidences, test_size=0.2, random_state=42
        )
        
        self.models['confidence_regressor'] = RandomForestRegressor(
            n_estimators=100, random_state=42
        )
        self.models['confidence_regressor'].fit(X_train, y_conf_train)
        
        y_conf_pred = self.models['confidence_regressor'].predict(X_test)
        conf_mse = np.mean((y_conf_test - y_conf_pred) ** 2)
        print(f"     신뢰도 회귀 MSE: {conf_mse:.3f}")
        
        # 4. 구간 분류기 훈련
        print("   - 구간 분류기 훈련...")
        section_encoder = {'intro': 0, 'verse1': 1, 'verse2': 2}
        y_sections_encoded = np.array([section_encoder.get(s, 0) for s in y_sections])
        
        X_train_sec, X_test_sec, y_sec_train, y_sec_test = train_test_split(
            X, y_sections_encoded, test_size=0.2, random_state=42
        )
        
        self.models['section_classifier'] = RandomForestClassifier(
            n_estimators=100, random_state=42
        )
        self.models['section_classifier'].fit(X_train_sec, y_sec_train)
        
        y_sec_pred = self.models['section_classifier'].predict(X_test_sec)
        sec_accuracy = accuracy_score(y_sec_test, y_sec_pred)
        print(f"     구간 분류 정확도: {sec_accuracy:.3f}")
        
        print("✅ 머신러닝 모델 훈련 완료")
    
    def predict_sync(self, audio_features: Dict, sheet_features: Dict) -> Dict:
        """머신러닝 모델을 사용한 동기화 예측"""
        # 특성 결합
        combined_features = {**audio_features, **sheet_features}
        
        # 특성 벡터로 변환
        feature_names = list(combined_features.keys())
        X = np.array([[combined_features[feature] for feature in feature_names]])
        
        # 스케일링
        X_scaled = self.scalers['combined_features'].transform(X)
        
        # 예측
        predictions = {}
        
        if self.models['sync_classifier']:
            sync_prob = self.models['sync_classifier'].predict_proba(X_scaled)[0]
            predictions['is_sync'] = sync_prob[1] > 0.5
            predictions['sync_confidence'] = sync_prob[1]
        
        if self.models['position_regressor'] and predictions.get('is_sync'):
            position_pred = self.models['position_regressor'].predict(X_scaled)[0]
            predictions['predicted_position'] = int(round(position_pred))
        
        if self.models['confidence_regressor']:
            confidence_pred = self.models['confidence_regressor'].predict(X_scaled)[0]
            predictions['ml_confidence'] = max(0, min(1, confidence_pred))
        
        if self.models['section_classifier']:
            section_pred = self.models['section_classifier'].predict(X_scaled)[0]
            section_names = ['intro', 'verse1', 'verse2']
            predictions['predicted_section'] = section_names[section_pred]
        
        return predictions
    
    def ml_sync_monitoring(self, duration: float = 60.0):
        """머신러닝 기반 동기화 모니터링"""
        print(f"🤖 ML 기반 동기화 모니터링 시작 (최대 {duration}초)...")
        
        # 훈련 데이터 생성 및 모델 훈련
        self.create_training_data()
        self.train_models()
        
        # 오디오 분석
        audio_analysis = self.audio_processor.analyze_full_audio()
        audio_results = audio_analysis['detailed_results']
        
        notes = self.sheet_music_data['notes']
        sync_results = []
        
        for result in audio_results[:int(duration * 10)]:
            current_time = result['time']
            current_pitch = result['pitch']
            
            # 오디오 특성 추출
            audio_features = {
                'pitch_mean': current_pitch,
                'pitch_std': 0,
                'volume_mean': result['volume'],
                'volume_std': 0,
                'tempo': 120,
                'onset_count': 0,
                'spectral_centroid_mean': current_pitch * 2,
            }
            
            best_match = None
            best_score = 0.0
            
            # 각 악보 음표와 ML 매칭 시도
            for note in notes:
                context_notes = notes[max(0, note['index']-2):min(len(notes), note['index']+3)]
                sheet_features = self.extract_sheet_features(note, context_notes)
                
                # ML 예측
                predictions = self.predict_sync(audio_features, sheet_features)
                
                if predictions.get('is_sync') and predictions.get('sync_confidence', 0) > best_score:
                    best_score = predictions['sync_confidence']
                    best_match = {
                        'note': note,
                        'predictions': predictions
                    }
            
            # 동기화 상태 업데이트
            if best_match and best_score > 0.5:
                self.sync_state['current_position'] = best_match['note']['index']
                self.sync_state['sync_confidence'] = best_score
                self.sync_state['is_synced'] = True
                self.sync_state['current_section'] = best_match['predictions'].get('predicted_section', 'unknown')
            else:
                self.sync_state['sync_confidence'] = best_score
                self.sync_state['is_synced'] = False
            
            # 결과 저장
            sync_results.append({
                'time': current_time,
                'audio_pitch': current_pitch,
                'best_match': best_match,
                'sync_state': self.sync_state.copy()
            })
            
            # 진행 상황 출력
            if len(sync_results) % 100 == 0:
                section = self.sync_state['current_section']
                print(f"⏱️ {current_time:.1f}초 [{section}]: "
                      f"음높이={current_pitch:.1f}Hz, "
                      f"ML신뢰도={best_score:.3f}, "
                      f"동기화={'✅' if self.sync_state['is_synced'] else '❌'}")
        
        return sync_results
    
    def save_models(self, model_dir: str = 'ml_models'):
        """훈련된 모델 저장"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            if model is not None:
                joblib.dump(model, f'{model_dir}/{model_name}.pkl')
        
        for scaler_name, scaler in self.scalers.items():
            joblib.dump(scaler, f'{model_dir}/{scaler_name}.pkl')
        
        print(f"✅ 모델 저장 완료: {model_dir}/")
    
    def load_models(self, model_dir: str = 'ml_models'):
        """저장된 모델 로드"""
        import os
        if not os.path.exists(model_dir):
            print(f"❌ 모델 디렉토리가 없습니다: {model_dir}")
            return False
        
        for model_name in self.models.keys():
            model_path = f'{model_dir}/{model_name}.pkl'
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
        
        for scaler_name in self.scalers.keys():
            scaler_path = f'{model_dir}/{scaler_name}.pkl'
            if os.path.exists(scaler_path):
                self.scalers[scaler_name] = joblib.load(scaler_path)
        
        print(f"✅ 모델 로드 완료: {model_dir}/")
        return True
    
    def convert_notes_to_timeline(self):
        """음표 데이터를 시간축으로 변환"""
        if not self.sheet_music_data or 'notes' not in self.sheet_music_data:
            return
        
        notes = self.sheet_music_data['notes']
        tempo = 120.0  # 기본 템포
        
        current_time = 0.0
        for i, note in enumerate(notes):
            note['start_time'] = current_time
            note_duration = note.get('duration', 1.0)
            note['end_time'] = current_time + (note_duration * 60.0 / tempo)
            current_time = note['end_time']
            note['index'] = i
    
    def note_name_to_frequency(self, note_name: str) -> float:
        """음표명을 주파수로 변환"""
        if note_name == "Silence":
            return 0.0
        
        note_map = {
            'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4,
            'F#': -3, 'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2
        }
        
        try:
            note = note_name[:-1]
            octave = int(note_name[-1])
            semitones = note_map[note] + (octave - 4) * 12
            frequency = 440.0 * (2 ** (semitones / 12))
            return frequency
        except:
            return 0.0
    
    def visualize_ml_sync_results(self, sync_results: List[Dict]):
        """ML 동기화 결과 시각화"""
        print("📊 ML 동기화 결과 시각화...")
        
        times = [r['time'] for r in sync_results]
        audio_pitches = [r['audio_pitch'] for r in sync_results]
        ml_confidences = [r['sync_state']['sync_confidence'] for r in sync_results]
        sync_status = [1 if r['sync_state']['is_synced'] else 0 for r in sync_results]
        sections = [r['sync_state']['current_section'] for r in sync_results]
        
        # 그래프 생성
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))
        
        # 1. 오디오 음높이
        axes[0].plot(times, audio_pitches, 'b-', alpha=0.7, label='Audio Pitch')
        axes[0].set_title('Audio Pitch Over Time (ML Enhanced)')
        axes[0].set_ylabel('Frequency (Hz)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. ML 신뢰도
        axes[1].plot(times, ml_confidences, 'g-', alpha=0.7)
        axes[1].axhline(y=0.5, color='orange', linestyle='--', label='ML Threshold')
        axes[1].set_title('ML Sync Confidence')
        axes[1].set_ylabel('Confidence')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 구간 분류
        section_colors = {'intro': 'red', 'verse1': 'blue', 'verse2': 'green', 'unknown': 'gray'}
        for i, section in enumerate(sections):
            if section in section_colors:
                axes[2].scatter(times[i], 1, c=section_colors[section], s=10, alpha=0.7)
        axes[2].set_title('ML Section Classification')
        axes[2].set_ylabel('Section')
        axes[2].set_ylim(0.5, 1.5)
        axes[2].grid(True, alpha=0.3)
        
        # 4. 동기화 상태
        axes[3].plot(times, sync_status, 'r-', alpha=0.7)
        axes[3].set_title('ML Sync Status')
        axes[3].set_ylabel('Synced (1/0)')
        axes[3].set_xlabel('Time (seconds)')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ml_sync_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ ML 시각화 완료: ml_sync_analysis.png")
    
    def get_ml_sync_statistics(self, sync_results: List[Dict]) -> Dict:
        """ML 동기화 통계 계산"""
        total_samples = len(sync_results)
        synced_samples = sum(1 for r in sync_results if r['sync_state']['is_synced'])
        
        avg_confidence = np.mean([r['sync_state']['sync_confidence'] for r in sync_results])
        max_confidence = max([r['sync_state']['sync_confidence'] for r in sync_results])
        
        # 구간별 통계
        section_stats = {}
        for section in set(r['sync_state']['current_section'] for r in sync_results):
            section_results = [r for r in sync_results if r['sync_state']['current_section'] == section]
            section_synced = sum(1 for r in section_results if r['sync_state']['is_synced'])
            section_stats[section] = {
                'total_samples': len(section_results),
                'synced_samples': section_synced,
                'sync_rate': section_synced / len(section_results) if section_results else 0,
                'avg_confidence': np.mean([r['sync_state']['sync_confidence'] for r in section_results])
            }
        
        return {
            'total_samples': total_samples,
            'synced_samples': synced_samples,
            'overall_sync_rate': synced_samples / total_samples if total_samples > 0 else 0,
            'avg_ml_confidence': avg_confidence,
            'max_ml_confidence': max_confidence,
            'section_statistics': section_stats
        }

def main():
    """메인 함수"""
    print("🤖 머신러닝 기반 악보-연주 싱크 시스템 시작")
    
    # 파일 경로
    sheet_music_path = "sheet music_1.png"
    audio_file_path = "주 품에 품으소서.mp3"
    
    try:
        # ML 싱크 시스템 초기화
        ml_sync = MLSyncSystem(sheet_music_path, audio_file_path)
        
        # ML 기반 동기화 모니터링 실행
        sync_results = ml_sync.ml_sync_monitoring(duration=60.0)
        
        # 통계 계산 및 출력
        stats = ml_sync.get_ml_sync_statistics(sync_results)
        print(f"\n📊 ML 동기화 통계:")
        print(f"   - 총 샘플 수: {stats['total_samples']}")
        print(f"   - 동기화된 샘플: {stats['synced_samples']}")
        print(f"   - 전체 동기화율: {stats['overall_sync_rate']:.2%}")
        print(f"   - 평균 ML 신뢰도: {stats['avg_ml_confidence']:.3f}")
        print(f"   - 최대 ML 신뢰도: {stats['max_ml_confidence']:.3f}")
        
        print(f"\n📈 구간별 ML 동기화율:")
        for section, section_stats in stats['section_statistics'].items():
            print(f"   - {section}: {section_stats['sync_rate']:.2%} ({section_stats['synced_samples']}/{section_stats['total_samples']})")
        
        # 결과 시각화
        ml_sync.visualize_ml_sync_results(sync_results)
        
        # 모델 저장
        ml_sync.save_models()
        
        print("\n✅ 머신러닝 기반 악보-연주 싱크 시스템 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
