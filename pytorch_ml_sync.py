"""
PyTorch + MPS 기반 머신러닝 악보-연주 싱크 시스템
Apple Silicon GPU 가속을 활용한 고성능 동기화 시스템
"""

import json
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from collections import deque
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from improved_omr import ImprovedOMR
from real_time_audio_processor import RealTimeAudioProcessor

class SyncDataset(Dataset):
    """동기화 데이터셋"""
    def __init__(self, features, labels, positions, confidences, sections):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.positions = torch.FloatTensor(positions)
        self.confidences = torch.FloatTensor(confidences)
        self.sections = torch.LongTensor(sections)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'sync_label': self.labels[idx],
            'position': self.positions[idx],
            'confidence': self.confidences[idx],
            'section': self.sections[idx]
        }

class SyncNet(nn.Module):
    """동기화 신경망 모델"""
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], num_sections=3):
        super(SyncNet, self).__init__()
        
        # 공통 특성 추출기
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 동기화 분류기
        self.sync_classifier = nn.Sequential(
            nn.Linear(prev_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)  # 동기화/비동기화
        )
        
        # 위치 회귀기
        self.position_regressor = nn.Sequential(
            nn.Linear(prev_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        # 신뢰도 회귀기
        self.confidence_regressor = nn.Sequential(
            nn.Linear(prev_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0-1 범위
        )
        
        # 구간 분류기
        self.section_classifier = nn.Sequential(
            nn.Linear(prev_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_sections)  # intro, verse1, verse2
        )
    
    def forward(self, x):
        # 공통 특성 추출
        features = self.feature_extractor(x)
        
        # 각 태스크별 출력
        sync_output = self.sync_classifier(features)
        position_output = self.position_regressor(features)
        confidence_output = self.confidence_regressor(features)
        section_output = self.section_classifier(features)
        
        return {
            'sync': sync_output,
            'position': position_output,
            'confidence': confidence_output,
            'section': section_output
        }

class PyTorchMLSyncSystem:
    def __init__(self, sheet_music_path: str, audio_file_path: str):
        """
        PyTorch + MPS 기반 싱크 시스템 초기화
        
        Args:
            sheet_music_path: 악보 이미지 경로
            audio_file_path: 오디오 파일 경로
        """
        self.sheet_music_path = sheet_music_path
        self.audio_file_path = audio_file_path
        
        # 디바이스 설정 (MPS 우선, 없으면 CPU)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🚀 MPS (Apple Silicon GPU) 가속 활성화")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("🚀 CUDA GPU 가속 활성화")
        else:
            self.device = torch.device("cpu")
            print("💻 CPU 모드로 실행")
        
        # OMR 시스템 초기화
        print("🎼 PyTorch ML 기반 OMR 시스템 초기화...")
        self.omr = ImprovedOMR()
        self.sheet_music_data = None
        
        # 오디오 처리 시스템 초기화
        print("🎵 오디오 처리 시스템 초기화...")
        self.audio_processor = RealTimeAudioProcessor(audio_file_path)
        
        # 모델 초기화
        self.model = None
        self.optimizer = None
        self.criterion = {
            'sync': nn.CrossEntropyLoss(),
            'position': nn.MSELoss(),
            'confidence': nn.MSELoss(),
            'section': nn.CrossEntropyLoss()
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
            'ml_predictions': deque(maxlen=10)
        }
        
        print("✅ PyTorch ML 기반 악보-연주 싱크 시스템 초기화 완료")
    
    def extract_enhanced_features(self, audio_chunk: np.ndarray, note: Dict, context_notes: List[Dict], sample_rate: int = 22050) -> Dict:
        """향상된 특성 추출"""
        features = {}
        
        # 1. 오디오 특성
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_chunk, fmin=80, fmax=1000, sr=sample_rate
        )
        
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) > 0:
            features['audio_pitch_mean'] = np.mean(valid_f0)
            features['audio_pitch_std'] = np.std(valid_f0)
            features['audio_pitch_min'] = np.min(valid_f0)
            features['audio_pitch_max'] = np.max(valid_f0)
            features['audio_pitch_range'] = features['audio_pitch_max'] - features['audio_pitch_min']
            features['audio_pitch_median'] = np.median(valid_f0)
        else:
            features.update({
                'audio_pitch_mean': 0, 'audio_pitch_std': 0, 'audio_pitch_min': 0,
                'audio_pitch_max': 0, 'audio_pitch_range': 0, 'audio_pitch_median': 0
            })
        
        # 볼륨 특성
        rms = librosa.feature.rms(y=audio_chunk)[0]
        features['audio_volume_mean'] = np.mean(rms)
        features['audio_volume_std'] = np.std(rms)
        features['audio_volume_max'] = np.max(rms)
        
        # 스펙트럴 특성
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_chunk, sr=sample_rate)[0]
        features['audio_spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['audio_spectral_centroid_std'] = np.std(spectral_centroids)
        
        # MFCC 특성 (음색)
        mfccs = librosa.feature.mfcc(y=audio_chunk, sr=sample_rate, n_mfcc=13)
        for i in range(13):
            features[f'audio_mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'audio_mfcc_{i}_std'] = np.std(mfccs[i])
        
        # 리듬 특성
        onset_frames = librosa.onset.onset_detect(y=audio_chunk, sr=sample_rate)
        features['audio_onset_count'] = len(onset_frames)
        if len(onset_frames) > 1:
            onset_intervals = np.diff(onset_frames)
            features['audio_onset_interval_mean'] = np.mean(onset_intervals)
            features['audio_onset_interval_std'] = np.std(onset_intervals)
        else:
            features['audio_onset_interval_mean'] = 0
            features['audio_onset_interval_std'] = 0
        
        # 템포 특성
        tempo, beats = librosa.beat.beat_track(y=audio_chunk, sr=sample_rate)
        features['audio_tempo'] = tempo
        features['audio_beat_count'] = len(beats)
        
        # 2. 악보 특성
        note_freq = self.note_name_to_frequency(note['pitch'])
        features['sheet_note_frequency'] = note_freq
        features['sheet_note_duration'] = note.get('duration', 1.0)
        features['sheet_note_area'] = note.get('area', 0)
        features['sheet_note_aspect_ratio'] = note.get('aspect_ratio', 1.0)
        features['sheet_note_position'] = note.get('index', 0)
        
        # 음표 타입 (원핫 인코딩)
        note_types = ['whole', 'half', 'quarter', 'eighth', 'sixteenth']
        for note_type in note_types:
            features[f'sheet_note_type_{note_type}'] = 1 if note.get('type') == note_type else 0
        
        # 3. 컨텍스트 특성
        if len(context_notes) > 0:
            context_freqs = [self.note_name_to_frequency(n['pitch']) for n in context_notes]
            context_freqs = [f for f in context_freqs if f > 0]
            
            if context_freqs:
                features['sheet_context_pitch_mean'] = np.mean(context_freqs)
                features['sheet_context_pitch_std'] = np.std(context_freqs)
                features['sheet_context_pitch_range'] = np.max(context_freqs) - np.min(context_freqs)
                features['sheet_context_pitch_median'] = np.median(context_freqs)
            else:
                features.update({
                    'sheet_context_pitch_mean': 0, 'sheet_context_pitch_std': 0,
                    'sheet_context_pitch_range': 0, 'sheet_context_pitch_median': 0
                })
            
            # 음정 관계
            if len(context_freqs) > 1:
                intervals = []
                for i in range(1, len(context_freqs)):
                    interval = 1200 * np.log2(context_freqs[i] / context_freqs[i-1])
                    intervals.append(interval)
                features['sheet_interval_mean'] = np.mean(intervals)
                features['sheet_interval_std'] = np.std(intervals)
                features['sheet_interval_count'] = len(intervals)
            else:
                features['sheet_interval_mean'] = 0
                features['sheet_interval_std'] = 0
                features['sheet_interval_count'] = 0
        
        # 4. 상대적 특성
        if features['audio_pitch_mean'] > 0 and features['sheet_note_frequency'] > 0:
            pitch_diff = abs(features['audio_pitch_mean'] - features['sheet_note_frequency'])
            features['pitch_difference'] = pitch_diff
            features['pitch_ratio'] = features['audio_pitch_mean'] / features['sheet_note_frequency']
            features['pitch_cents_diff'] = 1200 * np.log2(features['audio_pitch_mean'] / features['sheet_note_frequency'])
        else:
            features['pitch_difference'] = float('inf')
            features['pitch_ratio'] = 1.0
            features['pitch_cents_diff'] = 0
        
        return features
    
    def create_training_data(self):
        """훈련 데이터 생성"""
        print("📊 PyTorch 훈련 데이터 생성 중...")
        
        # 악보 데이터 로드
        self.sheet_music_data = self.omr.process_sheet_music_improved(self.sheet_music_path)
        
        # 오디오 분석
        audio_analysis = self.audio_processor.analyze_full_audio()
        audio_results = audio_analysis['detailed_results']
        
        # 시간축 변환
        self.convert_notes_to_timeline()
        
        notes = self.sheet_music_data['notes']
        
        # 샘플링을 통한 데이터 크기 조절 (메모리 효율성)
        max_samples = 50000  # 최대 5만 샘플
        audio_indices = np.random.choice(len(audio_results), min(max_samples, len(audio_results)), replace=False)
        note_indices = np.random.choice(len(notes), min(max_samples, len(notes)), replace=False)
        
        print(f"   - 오디오 샘플: {len(audio_indices)}개")
        print(f"   - 악보 샘플: {len(note_indices)}개")
        
        # 각 오디오-악보 쌍에 대해 훈련 데이터 생성
        print("   - 오디오-악보 쌍 매칭 중...")
        for i, audio_idx in enumerate(tqdm(audio_indices, desc="오디오 샘플 처리")):
            audio_result = audio_results[audio_idx]
            current_time = audio_result['time']
            current_pitch = audio_result['pitch']
            
            # 오디오 청크 시뮬레이션 (실제로는 audio_chunk가 필요)
            audio_chunk = np.random.normal(0, 0.1, 1024)  # 더미 데이터
            
            for j, note_idx in enumerate(note_indices):
                note = notes[note_idx]
                context_notes = notes[max(0, note_idx-2):min(len(notes), note_idx+3)]
                
                # 향상된 특성 추출
                features = self.extract_enhanced_features(audio_chunk, note, context_notes)
                
                # 라벨 생성 (개선된 휴리스틱)
                time_diff = abs(current_time - note['start_time'])
                pitch_diff = abs(current_pitch - features['sheet_note_frequency'])
                
                # 더 정교한 매칭 조건
                is_match = (time_diff < 8.0 and pitch_diff < 150 and 
                           features['pitch_cents_diff'] < 200)  # 8초, 150Hz, 200센트 허용
                
                confidence = max(0, 1.0 - (time_diff / 8.0) - (pitch_diff / 150) - (abs(features['pitch_cents_diff']) / 200))
                
                # 구간 라벨
                if current_time < 15:
                    section = 0  # intro
                elif current_time < 150:
                    section = 1  # verse1
                else:
                    section = 2  # verse2
                
                # 훈련 데이터에 추가
                self.training_data['features'].append(features)
                self.training_data['labels'].append(1 if is_match else 0)
                self.training_data['positions'].append(note_idx)
                self.training_data['confidences'].append(confidence)
                self.training_data['sections'].append(section)
        
        print(f"✅ PyTorch 훈련 데이터 생성 완료: {len(self.training_data['features'])}개 샘플")
    
    def prepare_pytorch_data(self) -> Tuple[DataLoader, DataLoader]:
        """PyTorch 데이터로더 준비"""
        if not self.training_data['features']:
            raise ValueError("훈련 데이터가 없습니다.")
        
        # 특성을 벡터로 변환
        feature_names = list(self.training_data['features'][0].keys())
        X = np.array([[sample[feature] for feature in feature_names] 
                     for sample in self.training_data['features']])
        
        # 라벨들
        y_sync = np.array(self.training_data['labels'])
        y_positions = np.array(self.training_data['positions'])
        y_confidences = np.array(self.training_data['confidences'])
        y_sections = np.array(self.training_data['sections'])
        
        # 특성 정규화
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1  # 표준편차가 0인 경우 방지
        X_normalized = (X - X_mean) / X_std
        
        # 데이터셋 생성
        dataset = SyncDataset(X_normalized, y_sync, y_positions, y_confidences, y_sections)
        
        # 훈련/검증 분할
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # 데이터로더 생성
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
        
        return train_loader, val_loader, feature_names, X_mean, X_std
    
    def train_pytorch_model(self, epochs: int = 50):
        """PyTorch 모델 훈련"""
        print("🤖 PyTorch 모델 훈련 시작...")
        
        # 데이터 준비
        train_loader, val_loader, feature_names, X_mean, X_std = self.prepare_pytorch_data()
        
        # 모델 초기화
        input_size = len(feature_names)
        self.model = SyncNet(input_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # 학습률 스케줄러
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)
        
        # 훈련 루프
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        # Epoch 진행바
        epoch_pbar = tqdm(range(epochs), desc="모델 훈련", unit="epoch")
        
        for epoch in epoch_pbar:
            # 훈련 모드
            self.model.train()
            train_loss = 0.0
            
            # 훈련 배치 진행바
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - 훈련", leave=False)
            
            for batch in train_pbar:
                # 데이터를 디바이스로 이동
                features = batch['features'].to(self.device)
                sync_labels = batch['sync_label'].to(self.device)
                positions = batch['position'].to(self.device)
                confidences = batch['confidence'].to(self.device)
                sections = batch['section'].to(self.device)
                
                # 순전파
                outputs = self.model(features)
                
                # 손실 계산
                sync_loss = self.criterion['sync'](outputs['sync'], sync_labels)
                position_loss = self.criterion['position'](outputs['position'].squeeze(), positions)
                confidence_loss = self.criterion['confidence'](outputs['confidence'].squeeze(), confidences)
                section_loss = self.criterion['section'](outputs['section'], sections)
                
                # 가중 합계 손실
                total_loss = (sync_loss + 0.5 * position_loss + 0.3 * confidence_loss + 0.2 * section_loss)
                
                # 역전파
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                train_loss += total_loss.item()
                
                # 훈련 진행바 업데이트
                train_pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Sync': f'{sync_loss.item():.4f}',
                    'Pos': f'{position_loss.item():.4f}'
                })
            
            # 검증 모드
            self.model.eval()
            val_loss = 0.0
            val_sync_acc = 0.0
            val_section_acc = 0.0
            
            # 검증 배치 진행바
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - 검증", leave=False)
            
            with torch.no_grad():
                for batch in val_pbar:
                    features = batch['features'].to(self.device)
                    sync_labels = batch['sync_label'].to(self.device)
                    positions = batch['position'].to(self.device)
                    confidences = batch['confidence'].to(self.device)
                    sections = batch['section'].to(self.device)
                    
                    outputs = self.model(features)
                    
                    # 손실 계산
                    sync_loss = self.criterion['sync'](outputs['sync'], sync_labels)
                    position_loss = self.criterion['position'](outputs['position'].squeeze(), positions)
                    confidence_loss = self.criterion['confidence'](outputs['confidence'].squeeze(), confidences)
                    section_loss = self.criterion['section'](outputs['section'], sections)
                    
                    total_loss = (sync_loss + 0.5 * position_loss + 0.3 * confidence_loss + 0.2 * section_loss)
                    val_loss += total_loss.item()
                    
                    # 정확도 계산
                    sync_pred = torch.argmax(outputs['sync'], dim=1)
                    section_pred = torch.argmax(outputs['section'], dim=1)
                    
                    val_sync_acc += (sync_pred == sync_labels).float().mean().item()
                    val_section_acc += (section_pred == sections).float().mean().item()
                    
                    # 검증 진행바 업데이트
                    val_pbar.set_postfix({
                        'Val Loss': f'{total_loss.item():.4f}',
                        'Sync Acc': f'{(sync_pred == sync_labels).float().mean().item():.3f}'
                    })
            
            # 평균 손실 계산
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_sync_acc /= len(val_loader)
            val_section_acc /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # 학습률 스케줄링
            scheduler.step(val_loss)
            
            # 최고 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_sync_model.pth')
            
            # Epoch 진행바 업데이트
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'Sync Acc': f'{val_sync_acc:.3f}',
                'Section Acc': f'{val_section_acc:.3f}',
                'Best Val': f'{best_val_loss:.4f}'
            })
        
        print("✅ PyTorch 모델 훈련 완료")
        
        # 훈련 곡선 시각화
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(train_losses[-20:], label='Train Loss (Last 20)')
        plt.plot(val_losses[-20:], label='Validation Loss (Last 20)')
        plt.title('Recent Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('pytorch_training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 최고 모델 로드
        self.model.load_state_dict(torch.load('best_sync_model.pth'))
        
        return feature_names, X_mean, X_std
    
    def predict_with_pytorch(self, audio_features: Dict, sheet_features: Dict, feature_names: List[str], X_mean: np.ndarray, X_std: np.ndarray) -> Dict:
        """PyTorch 모델을 사용한 예측"""
        # 특성 결합 및 정규화
        combined_features = {**audio_features, **sheet_features}
        X = np.array([[combined_features[feature] for feature in feature_names]])
        X_normalized = (X - X_mean) / X_std
        
        # 텐서로 변환
        X_tensor = torch.FloatTensor(X_normalized).to(self.device)
        
        # 예측
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            # 확률 계산
            sync_probs = F.softmax(outputs['sync'], dim=1)
            section_probs = F.softmax(outputs['section'], dim=1)
            
            predictions = {
                'is_sync': sync_probs[0, 1].item() > 0.5,
                'sync_confidence': sync_probs[0, 1].item(),
                'predicted_position': int(round(outputs['position'][0, 0].item())),
                'ml_confidence': outputs['confidence'][0, 0].item(),
                'predicted_section': torch.argmax(section_probs, dim=1)[0].item()
            }
        
        return predictions
    
    def pytorch_sync_monitoring(self, duration: float = 60.0):
        """PyTorch 기반 동기화 모니터링"""
        print(f"🤖 PyTorch 기반 동기화 모니터링 시작 (최대 {duration}초)...")
        
        # 훈련 데이터 생성 및 모델 훈련
        self.create_training_data()
        feature_names, X_mean, X_std = self.train_pytorch_model(epochs=30)
        
        # 오디오 분석
        audio_analysis = self.audio_processor.analyze_full_audio()
        audio_results = audio_analysis['detailed_results']
        
        notes = self.sheet_music_data['notes']
        sync_results = []
        
        # 동기화 모니터링 진행바
        monitor_pbar = tqdm(audio_results[:int(duration * 10)], desc="동기화 모니터링", unit="sample")
        
        for result in monitor_pbar:
            current_time = result['time']
            current_pitch = result['pitch']
            
            # 오디오 특성 추출
            audio_chunk = np.random.normal(0, 0.1, 1024)  # 더미 데이터
            audio_features = self.extract_enhanced_features(audio_chunk, {'pitch': 'C4'}, [])  # 더미 악보
            
            best_match = None
            best_score = 0.0
            
            # 각 악보 음표와 PyTorch 매칭 시도
            for note in notes:
                context_notes = notes[max(0, note['index']-2):min(len(notes), note['index']+3)]
                sheet_features = self.extract_enhanced_features(audio_chunk, note, context_notes)
                
                # PyTorch 예측
                predictions = self.predict_with_pytorch(audio_features, sheet_features, feature_names, X_mean, X_std)
                
                if predictions['is_sync'] and predictions['sync_confidence'] > best_score:
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
                section_names = ['intro', 'verse1', 'verse2']
                self.sync_state['current_section'] = section_names[best_match['predictions']['predicted_section']]
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
            
            # 진행바 업데이트
            section = self.sync_state['current_section']
            monitor_pbar.set_postfix({
                'Time': f'{current_time:.1f}s',
                'Pitch': f'{current_pitch:.1f}Hz',
                'Section': section,
                'Confidence': f'{best_score:.3f}',
                'Synced': '✅' if self.sync_state['is_synced'] else '❌'
            })
        
        return sync_results
    
    def convert_notes_to_timeline(self):
        """음표 데이터를 시간축으로 변환"""
        if not self.sheet_music_data or 'notes' not in self.sheet_music_data:
            return
        
        notes = self.sheet_music_data['notes']
        tempo = 120.0
        
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
    
    def visualize_pytorch_sync_results(self, sync_results: List[Dict]):
        """PyTorch 동기화 결과 시각화"""
        print("📊 PyTorch 동기화 결과 시각화...")
        
        times = [r['time'] for r in sync_results]
        audio_pitches = [r['audio_pitch'] for r in sync_results]
        pytorch_confidences = [r['sync_state']['sync_confidence'] for r in sync_results]
        sync_status = [1 if r['sync_state']['is_synced'] else 0 for r in sync_results]
        sections = [r['sync_state']['current_section'] for r in sync_results]
        
        # 그래프 생성
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))
        
        # 1. 오디오 음높이
        axes[0].plot(times, audio_pitches, 'b-', alpha=0.7, label='Audio Pitch')
        axes[0].set_title('Audio Pitch Over Time (PyTorch + MPS)')
        axes[0].set_ylabel('Frequency (Hz)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. PyTorch 신뢰도
        axes[1].plot(times, pytorch_confidences, 'g-', alpha=0.7)
        axes[1].axhline(y=0.5, color='orange', linestyle='--', label='PyTorch Threshold')
        axes[1].set_title('PyTorch Sync Confidence')
        axes[1].set_ylabel('Confidence')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 구간 분류
        section_colors = {'intro': 'red', 'verse1': 'blue', 'verse2': 'green', 'unknown': 'gray'}
        for i, section in enumerate(sections):
            if section in section_colors:
                axes[2].scatter(times[i], 1, c=section_colors[section], s=10, alpha=0.7)
        axes[2].set_title('PyTorch Section Classification')
        axes[2].set_ylabel('Section')
        axes[2].set_ylim(0.5, 1.5)
        axes[2].grid(True, alpha=0.3)
        
        # 4. 동기화 상태
        axes[3].plot(times, sync_status, 'r-', alpha=0.7)
        axes[3].set_title('PyTorch Sync Status')
        axes[3].set_ylabel('Synced (1/0)')
        axes[3].set_xlabel('Time (seconds)')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pytorch_sync_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ PyTorch 시각화 완료: pytorch_sync_analysis.png")
    
    def get_pytorch_sync_statistics(self, sync_results: List[Dict]) -> Dict:
        """PyTorch 동기화 통계 계산"""
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
            'avg_pytorch_confidence': avg_confidence,
            'max_pytorch_confidence': max_confidence,
            'section_statistics': section_stats,
            'device_used': str(self.device)
        }

def main():
    """메인 함수"""
    print("🤖 PyTorch + MPS 기반 악보-연주 싱크 시스템 시작")
    
    # 파일 경로
    sheet_music_path = "sheet music_1.png"
    audio_file_path = "주 품에 품으소서.mp3"
    
    try:
        # PyTorch ML 싱크 시스템 초기화
        pytorch_sync = PyTorchMLSyncSystem(sheet_music_path, audio_file_path)
        
        # PyTorch 기반 동기화 모니터링 실행
        sync_results = pytorch_sync.pytorch_sync_monitoring(duration=60.0)
        
        # 통계 계산 및 출력
        stats = pytorch_sync.get_pytorch_sync_statistics(sync_results)
        print(f"\n📊 PyTorch 동기화 통계:")
        print(f"   - 사용 디바이스: {stats['device_used']}")
        print(f"   - 총 샘플 수: {stats['total_samples']}")
        print(f"   - 동기화된 샘플: {stats['synced_samples']}")
        print(f"   - 전체 동기화율: {stats['overall_sync_rate']:.2%}")
        print(f"   - 평균 PyTorch 신뢰도: {stats['avg_pytorch_confidence']:.3f}")
        print(f"   - 최대 PyTorch 신뢰도: {stats['max_pytorch_confidence']:.3f}")
        
        print(f"\n📈 구간별 PyTorch 동기화율:")
        for section, section_stats in stats['section_statistics'].items():
            print(f"   - {section}: {section_stats['sync_rate']:.2%} ({section_stats['synced_samples']}/{section_stats['total_samples']})")
        
        # 결과 시각화
        pytorch_sync.visualize_pytorch_sync_results(sync_results)
        
        print("\n✅ PyTorch + MPS 기반 악보-연주 싱크 시스템 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
