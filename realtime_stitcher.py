#!/usr/bin/env python3
"""
실시간 360도 파노라마 스티칭 시스템 (하이브리드 v3 기반)
- ChArUco 왜곡 보정
- 45도 원형 배치 고정
- 멀티스레드 프레임 캡처
- 테스트 모드 + 실시간 모드
"""

import os
import glob
import time
import argparse
import threading
import queue
from collections import deque
import numpy as np
import cv2
import yaml


# stitcher_hybrid_v3.py의 핵심 함수들 임포트
from stitcher_hybrid_v3 import (
    load_calibration,
    undistort_image,
    warp_to_equirectangular,
    blend_images_multiband
)


class CameraCalibration:
    """캘리브레이션 데이터 관리"""
    
    def __init__(self, intrinsics, extrinsics):
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.undistort_maps = {}
    
    def prepare_undistort_maps(self, img_size):
        """왜곡 보정 맵 사전 계산 (속도 최적화)"""
        h, w = img_size
        
        for cam_idx, calib in self.intrinsics.items():
            K = calib['K']
            dist = calib['dist']
            
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
            
            map1, map2 = cv2.initUndistortRectifyMap(
                K, dist, None, new_K, (w, h), cv2.CV_16SC2
            )
            
            self.undistort_maps[cam_idx] = (map1, map2, new_K)
        
        print(f"✅ 왜곡 보정 맵 준비 완료")
    
    def undistort(self, cam_idx, img):
        """빠른 왜곡 보정 (사전 계산된 맵 사용)"""
        if cam_idx not in self.undistort_maps:
            return img, self.intrinsics[cam_idx]['K']
        
        map1, map2, new_K = self.undistort_maps[cam_idx]
        undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        return undistorted, new_K


class FrameCapture:
    """프레임 캡처 (테스트 모드 + 실시간 모드)"""
    
    def __init__(self, mode='test', input_dir=None, stream_urls=None, num_frames=10):
        self.mode = mode
        self.input_dir = input_dir
        self.stream_urls = stream_urls or {}
        self.num_frames = num_frames
        self.frame_queues = {i: queue.Queue(maxsize=2) for i in range(1, 9)}
        self.running = False
        self.threads = []
        
        # 테스트 모드: 이미지 파일 로드
        if mode == 'test':
            self.test_images = self._load_test_images(self.num_frames)
    
    def _load_test_images(self, num_frames=10):
        """테스트용 이미지 로드 (여러 프레임)"""
        images = {}
        
        for cam_idx in range(1, 9):
            cam_dir = os.path.join(self.input_dir, f'MyCam_{cam_idx:03d}')
            
            if not os.path.exists(cam_dir):
                continue
            
            img_files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))
            
            if img_files:
                # 처음 num_frames개 이미지 로드
                frames = []
                for img_file in img_files[:num_frames]:
                    img = cv2.imread(img_file)
                    if img is not None:
                        frames.append(img)
                
                if frames:
                    images[cam_idx] = frames
                    print(f"  카메라 {cam_idx}: {len(frames)}장 로드")
        
        print(f"✅ 테스트 이미지 로드: {len(images)}개 카메라")
        return images
    
    def start(self):
        """캡처 시작"""
        self.running = True
        
        if self.mode == 'test':
            # 테스트 모드: 반복해서 같은 이미지 제공
            thread = threading.Thread(target=self._test_capture_loop)
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        else:
            # 실시간 모드: 각 카메라별 스레드
            for cam_idx in range(1, 9):
                if cam_idx in self.stream_urls:
                    thread = threading.Thread(
                        target=self._stream_capture_loop,
                        args=(cam_idx,)
                    )
                    thread.daemon = True
                    thread.start()
                    self.threads.append(thread)
    
    def _test_capture_loop(self):
        """테스트 모드 루프 (순차 재생)"""
        frame_idx = 0
        
        while self.running:
            for cam_idx, frames in self.test_images.items():
                # 현재 프레임 인덱스에 해당하는 이미지
                img = frames[frame_idx % len(frames)]
                
                try:
                    self.frame_queues[cam_idx].put(img.copy(), block=False)
                except queue.Full:
                    pass
            
            frame_idx += 1
            time.sleep(0.033)  # 약 30 FPS로 제공
    
    def _stream_capture_loop(self, cam_idx):
        """실시간 스트리밍 캡처 루프"""
        url = self.stream_urls[cam_idx]
        cap = cv2.VideoCapture(url)
        
        if not cap.isOpened():
            print(f"❌ 카메라 {cam_idx} 연결 실패: {url}")
            return
        
        print(f"✅ 카메라 {cam_idx} 연결 성공")
        
        while self.running:
            ret, frame = cap.read()
            
            if not ret:
                print(f"⚠️ 카메라 {cam_idx} 프레임 읽기 실패")
                time.sleep(0.1)
                continue
            
            try:
                self.frame_queues[cam_idx].put(frame, block=False)
            except queue.Full:
                pass
        
        cap.release()
    
    def get_frames(self, timeout=1.0):
        """모든 카메라의 최신 프레임 가져오기"""
        frames = {}
        
        for cam_idx in range(1, 9):
            try:
                frame = self.frame_queues[cam_idx].get(timeout=timeout)
                frames[cam_idx] = frame
            except queue.Empty:
                pass
        
        return frames
    
    def stop(self):
        """캡처 중지"""
        self.running = False
        for thread in self.threads:
            thread.join(timeout=1.0)


class RealtimePanoramaStitcher:
    """실시간 파노라마 스티칭 (하이브리드 v3)"""
    
    def __init__(self, calibration, capture, output_width=8192, output_height=4096):
        self.calibration = calibration
        self.capture = capture
        self.output_width = output_width
        self.output_height = output_height
        
        # 성능 모니터링
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()
    
    def process_frame(self):
        """한 프레임 처리"""
        # 1. 프레임 가져오기
        frames = self.capture.get_frames(timeout=0.5)
        
        if len(frames) < 2:
            return None, 0.0
        
        # 2. 왜곡 보정 + 워핑
        warped_images = []
        
        for cam_idx in sorted(frames.keys()):
            img = frames[cam_idx]
            
            # 왜곡 보정
            img_undist, K = self.calibration.undistort(cam_idx, img)
            
            # 회전 행렬
            R = self.calibration.extrinsics[cam_idx]['R']
            
            # 시작 각도 (45도 간격)
            start_angle = (cam_idx - 1) * 45
            
            # 워핑
            warped, x_start, x_end = warp_to_equirectangular(
                img_undist, K, R,
                self.output_width, self.output_height,
                start_angle, fov_deg=50
            )
            
            warped_images.append((warped, x_start, x_end))
        
        # 3. 블렌딩
        start_time = time.time()
        panorama = blend_images_multiband(
            warped_images, None,
            self.output_width, self.output_height
        )
        blend_time = time.time() - start_time
        
        # 4. FPS 계산
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.fps_history.append(fps)
        self.last_time = current_time
        
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        return panorama, avg_fps
    
    def run(self, display=True, save_video=None):
        """실시간 스티칭 실행"""
        print("\n" + "="*60)
        print("실시간 파노라마 스티칭 시작 (하이브리드 v3)")
        print("="*60)
        print("종료: 'q' 키")
        print("="*60 + "\n")
        
        # 비디오 저장 설정
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                save_video, fourcc, 10.0,
                (self.output_width, self.output_height)
            )
        
        # 캡처 시작
        self.capture.start()
        
        try:
            frame_count = 0
            
            while True:
                panorama, fps = self.process_frame()
                
                if panorama is not None:
                    frame_count += 1
                    
                    # FPS 표시
                    cv2.putText(
                        panorama,
                        f"FPS: {fps:.1f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2
                    )
                    
                    # 화면 표시 (리사이즈)
                    if display:
                        display_width = 1920
                        display_height = int(panorama.shape[0] * display_width / panorama.shape[1])
                        display_img = cv2.resize(panorama, (display_width, display_height))
                        cv2.imshow('Real-time Panorama', display_img)
                    
                    # 비디오 저장
                    if video_writer:
                        video_writer.write(panorama)
                    
                    # 콘솔 출력
                    if frame_count % 10 == 0:
                        print(f"프레임 {frame_count}: {fps:.1f} FPS")
                
                # 키 입력 확인
                if display:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                else:
                    time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n\n중단됨")
        
        finally:
            # 정리
            self.capture.stop()
            
            if video_writer:
                video_writer.release()
            
            if display:
                cv2.destroyAllWindows()
            
            print(f"\n✅ 총 {frame_count}개 프레임 처리")
            if self.fps_history:
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                print(f"✅ 평균 FPS: {avg_fps:.1f}")


def main():
    parser = argparse.ArgumentParser(description='실시간 360도 파노라마 스티칭 (하이브리드 v3)')
    parser.add_argument('--mode', choices=['test', 'realtime'], default='test',
                        help='실행 모드 (test: 저장된 이미지, realtime: 스트리밍)')
    parser.add_argument('--intrinsics', required=True, help='내부 파라미터 파일')
    parser.add_argument('--extrinsics', required=True, help='외부 파라미터 파일')
    parser.add_argument('--input_dir', help='테스트 이미지 디렉토리 (test 모드)')
    parser.add_argument('--num_frames', type=int, default=10, help='테스트할 프레임 수 (test 모드)')
    parser.add_argument('--streams', help='스트리밍 URL 파일 (realtime 모드)')
    parser.add_argument('--output_width', type=int, default=8192, help='출력 너비')
    parser.add_argument('--output_height', type=int, default=4096, help='출력 높이')
    parser.add_argument('--save_video', help='비디오 저장 경로')
    parser.add_argument('--no_display', action='store_true', help='화면 표시 안 함')
    
    args = parser.parse_args()
    
    # 캘리브레이션 로드
    print("캘리브레이션 로드 중...")
    intrinsics, extrinsics = load_calibration(args.intrinsics, args.extrinsics)
    print(f"  내부 파라미터: {len(intrinsics)}개 카메라")
    print(f"  외부 파라미터: {len(extrinsics)}개 카메라")
    
    calibration = CameraCalibration(intrinsics, extrinsics)
    
    # 프레임 캡처 초기화
    if args.mode == 'test':
        if not args.input_dir:
            print("❌ 테스트 모드는 --input_dir 필요")
            return
        
        capture = FrameCapture(
            mode='test',
            input_dir=os.path.expanduser(args.input_dir),
            num_frames=args.num_frames
        )
        
        # 왜곡 보정 맵 준비
        if capture.test_images:
            sample_frames = next(iter(capture.test_images.values()))
            sample_img = sample_frames[0]
            calibration.prepare_undistort_maps(sample_img.shape[:2])
    
    else:  # realtime
        if not args.streams:
            print("❌ 실시간 모드는 --streams 필요")
            return
        
        # 스트리밍 URL 로드
        with open(args.streams, 'r') as f:
            stream_data = yaml.safe_load(f)
        
        capture = FrameCapture(
            mode='realtime',
            stream_urls=stream_data
        )
    
    # 스티칭 실행
    stitcher = RealtimePanoramaStitcher(
        calibration, capture,
        args.output_width, args.output_height
    )
    
    stitcher.run(
        display=not args.no_display,
        save_video=args.save_video
    )


if __name__ == '__main__':
    main()
