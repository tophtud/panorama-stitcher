#!/usr/bin/env python3
"""
8대 카메라 OpenCV Stitcher v6 (GPU 가속 + 카메라 순서 재배치)
- OpenCV Stitcher 완전 활용
- GPU 가속
- 카메라 순서 지정
- 최고 품질
"""

import os
import glob
import argparse
import time
import numpy as np
import cv2


def load_camera_images(input_dir, num_cameras=8, num_frames=10, crop_edges=None, camera_order=None):
    """
    각 카메라의 이미지 로드 + 순서 재배치
    """
    print("\n이미지 로드 중...")
    
    # 크롭 설정
    if crop_edges is None:
        crop_left = crop_right = crop_top = crop_bottom = 0
    elif isinstance(crop_edges, int):
        crop_left = crop_right = crop_top = crop_bottom = crop_edges
    else:
        crop_left, crop_right, crop_top, crop_bottom = crop_edges
    
    if any([crop_left, crop_right, crop_top, crop_bottom]):
        print(f"  가장자리 크롭: 좌={crop_left}, 우={crop_right}, 상={crop_top}, 하={crop_bottom} 픽셀")
    
    # 카메라 순서
    if camera_order is None:
        camera_order = list(range(1, num_cameras + 1))
    
    print(f"  카메라 순서: {camera_order}")
    
    all_images = []
    
    for cam_idx in camera_order:
        cam_dir = os.path.join(input_dir, f'MyCam_{cam_idx:03d}')
        
        if not os.path.exists(cam_dir):
            print(f"  ⚠️ 카메라 {cam_idx} 디렉토리 없음")
            continue
        
        img_files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))
        
        if not img_files:
            print(f"  ⚠️ 카메라 {cam_idx} 이미지 없음")
            continue
        
        # 처음 num_frames개 로드
        images = []
        for img_file in img_files[:num_frames]:
            img = cv2.imread(img_file)
            if img is not None:
                # 가장자리 크롭
                if crop_left > 0:
                    img[:, :crop_left] = 0
                if crop_right > 0:
                    img[:, -crop_right:] = 0
                if crop_top > 0:
                    img[:crop_top, :] = 0
                if crop_bottom > 0:
                    img[-crop_bottom:, :] = 0
                
                images.append(img)
        
        if images:
            all_images.append(images)
            print(f"  ✅ 카메라 {cam_idx}: {len(images)}장")
    
    print(f"\n✅ 총 {len(all_images)}개 카메라")
    
    return all_images


class GPUStitcherV6:
    """
    OpenCV Stitcher v6 (GPU 가속 + 카메라 순서 재배치)
    """
    
    def __init__(self, scale=1.0, try_use_gpu=True):
        self.scale = scale
        self.try_use_gpu = try_use_gpu
        self.stitcher = None
        self.is_initialized = False
    
    def calibrate(self, images):
        """
        OpenCV Stitcher로 캘리브레이션
        """
        print("\n" + "="*60)
        print(f"OpenCV Stitcher v6 (GPU 가속 + 카메라 순서)")
        print(f"스케일: {int(self.scale*100)}%")
        print(f"GPU 사용: {self.try_use_gpu}")
        print("="*60)
        
        # 리사이즈
        images_scaled = [
            cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            for img in images
        ]
        
        h, w = images_scaled[0].shape[:2]
        
        print(f"  입력: {len(images_scaled)}개 카메라")
        print(f"  크기: {w} x {h}")
        
        # OpenCV Stitcher 생성
        print("\n  1. OpenCV Stitcher 생성 중...")
        self.stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        
        # GPU 설정
        if self.try_use_gpu:
            try:
                # GPU 사용 시도
                self.stitcher.setPanoConfidenceThresh(1.0)
                print("    GPU 가속 활성화 시도")
            except:
                print("    GPU 사용 불가 - CPU 모드")
        
        # 고급 설정
        print("\n  2. 고급 설정 적용 중...")
        
        # 작업 해상도 (메가픽셀)
        # -1 = 원본 해상도 사용
        try:
            self.stitcher.setRegistrationResol(0.6)  # 등록 해상도
            self.stitcher.setSeamEstimationResol(0.1)  # Seam 추정 해상도
            self.stitcher.setCompositingResol(-1)  # 합성 해상도 (원본)
            self.stitcher.setPanoConfidenceThresh(0.5)  # 신뢰도 임계값 낮춤
            print("    등록 해상도: 0.6 MP")
            print("    Seam 해상도: 0.1 MP")
            print("    합성 해상도: 원본")
            print("    신뢰도 임계값: 0.5")
        except Exception as e:
            print(f"    설정 적용 실패: {e}")
        
        # 캘리브레이션
        print("\n  3. 변환 추정 중...")
        status = self.stitcher.estimateTransform(images_scaled)
        
        if status != cv2.Stitcher_OK:
            print(f"  ❌ 캘리브레이션 실패: {status}")
            if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
                print("     원인: 이미지 부족 또는 특징점 부족")
            elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
                print("     원인: 호모그래피 추정 실패")
            elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
                print("     원인: 카메라 파라미터 조정 실패")
            return None
        
        print(f"  ✅ 변환 추정 성공!")
        
        # 카메라 정보
        try:
            cameras = self.stitcher.cameras()
            print(f"  사용된 카메라 수: {len(cameras)}")
            if cameras:
                avg_focal = np.mean([cam.focal for cam in cameras])
                print(f"  평균 focal: {avg_focal:.1f}")
        except:
            pass
        
        # 기준 파노라마 생성
        print("\n  4. 기준 파노라마 생성 중...")
        status2, reference = self.stitcher.composePanorama()
        
        if status2 != cv2.Stitcher_OK:
            print(f"  ❌ 파노라마 생성 실패: {status2}")
            return None
        
        print(f"  ✅ 기준 파노라마 생성 성공!")
        print(f"  크기: {reference.shape[1]} x {reference.shape[0]}")
        
        self.is_initialized = True
        
        print("\n✅ 캘리브레이션 완료!")
        print("="*60)
        
        return reference
    
    def stitch(self, images):
        """
        이미지 스티칭
        """
        if not self.is_initialized:
            return None
        
        # 리사이즈
        images_scaled = [
            cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            for img in images
        ]
        
        # 스티칭 (캘리브레이션 재사용)
        status, panorama = self.stitcher.composePanorama(images_scaled)
        
        if status != cv2.Stitcher_OK:
            return None
        
        return panorama


def main():
    parser = argparse.ArgumentParser(
        description='8대 카메라 OpenCV Stitcher v6 (GPU 가속 + 카메라 순서)'
    )
    parser.add_argument('--mode', choices=['test', 'realtime'], default='test')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--num_frames', type=int, default=10)
    parser.add_argument('--reference_frame', type=int, default=0)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--camera_order', type=int, nargs='+', help='카메라 순서 (예: 5 4 3 2 1 8 7 6)')
    parser.add_argument('--try_use_gpu', action='store_true', help='GPU 사용 시도')
    parser.add_argument('--crop_edges', type=int, default=0)
    parser.add_argument('--crop_left', type=int, default=0)
    parser.add_argument('--crop_right', type=int, default=0)
    parser.add_argument('--crop_top', type=int, default=0)
    parser.add_argument('--crop_bottom', type=int, default=0)
    parser.add_argument('--save_video', help='비디오 저장 경로')
    parser.add_argument('--save_images', help='이미지 저장 디렉토리')
    parser.add_argument('--save_reference', help='기준 파노라마 저장 경로')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--loop', action='store_true')
    
    args = parser.parse_args()
    
    print("="*60)
    print("OpenCV Stitcher v6 (GPU 가속 + 카메라 순서)")
    print("="*60)
    print(f"모드: {args.mode}")
    print(f"기준 프레임: {args.reference_frame}")
    print(f"스케일: {args.scale * 100:.0f}%")
    print(f"GPU 사용: {args.try_use_gpu}")
    if args.camera_order:
        print(f"카메라 순서: {args.camera_order}")
    
    # 크롭 설정
    if args.crop_edges > 0:
        crop_config = args.crop_edges
    elif any([args.crop_left, args.crop_right, args.crop_top, args.crop_bottom]):
        crop_config = (args.crop_left, args.crop_right, args.crop_top, args.crop_bottom)
    else:
        crop_config = None
    
    # 이미지 로드
    all_images = load_camera_images(
        os.path.expanduser(args.input_dir),
        num_frames=args.num_frames,
        crop_edges=crop_config,
        camera_order=args.camera_order
    )
    
    if len(all_images) < 2:
        print("\n❌ 카메라가 부족합니다")
        return
    
    num_frames = len(all_images[0])
    
    print(f"\n프레임 수: {num_frames}")
    
    # 스티처 초기화
    stitcher = GPUStitcherV6(
        scale=args.scale,
        try_use_gpu=args.try_use_gpu
    )
    
    # 기준 프레임 추출
    ref_idx = min(args.reference_frame, num_frames - 1)
    ref_images = [cam_images[ref_idx] for cam_images in all_images]
    
    # 캘리브레이션
    reference = stitcher.calibrate(ref_images)
    
    if reference is None:
        print("\n❌ 캘리브레이션 실패")
        return
    
    if args.save_reference:
        cv2.imwrite(args.save_reference, reference)
        print(f"\n✅ 기준 파노라마 저장: {args.save_reference}")
    
    # 저장 설정
    video_writer = None
    
    if args.save_images:
        os.makedirs(args.save_images, exist_ok=True)
        print(f"\n이미지 저장 디렉토리: {args.save_images}")
    
    # 스티칭 루프
    print("\n" + "="*60)
    print("스티칭 시작 (q 키로 종료)")
    print("="*60)
    
    frame_idx = 0
    fps_list = []
    success_count = 0
    
    try:
        while True:
            start_time = time.time()
            
            # 현재 프레임 추출
            idx = frame_idx % num_frames
            images = [cam_images[idx] for cam_images in all_images]
            
            panorama = stitcher.stitch(images)
            
            if panorama is not None:
                success_count += 1
                
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                fps_list.append(fps)
                
                avg_fps = np.mean(fps_list[-30:])
                cv2.putText(
                    panorama,
                    f"FPS: {avg_fps:.1f} | Frame: {frame_idx+1}/{num_frames} | Stitcher v6",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )
                
                if video_writer is None and args.save_video:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(
                        args.save_video,
                        fourcc,
                        args.fps,
                        (panorama.shape[1], panorama.shape[0])
                    )
                    print(f"\n비디오 저장 시작: {args.save_video}")
                
                display_height = 600
                display_width = int(display_height * panorama.shape[1] / panorama.shape[0])
                display = cv2.resize(panorama, (display_width, display_height))
                cv2.imshow('Panorama', display)
                
                if video_writer:
                    video_writer.write(panorama)
                
                if args.save_images:
                    img_path = os.path.join(args.save_images, f'panorama_{frame_idx:04d}.jpg')
                    cv2.imwrite(img_path, panorama)
                
                if (frame_idx + 1) % 10 == 0:
                    print(f"  프레임 {frame_idx+1}/{num_frames} | FPS: {avg_fps:.1f}")
            
            else:
                print(f"  ⚠️ 프레임 {frame_idx+1} 스티칭 실패")
            
            wait_time = max(1, int(1000 / args.fps) - int((time.time() - start_time) * 1000))
            key = cv2.waitKey(wait_time)
            if key == ord('q'):
                break
            
            frame_idx += 1
            
            if frame_idx >= num_frames:
                if args.loop:
                    frame_idx = 0
                    print("\n반복 재생...")
                else:
                    break
    
    except KeyboardInterrupt:
        print("\n\n중단됨")
    
    finally:
        if video_writer:
            video_writer.release()
        
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("통계")
        print("="*60)
        print(f"성공: {success_count} 프레임")
        
        if fps_list:
            print(f"\n평균 FPS: {np.mean(fps_list):.2f}")
            print(f"최소 FPS: {np.min(fps_list):.2f}")
            print(f"최대 FPS: {np.max(fps_list):.2f}")
        
        if args.save_video and video_writer:
            print(f"\n✅ 비디오 저장: {args.save_video}")
        
        if args.save_images:
            print(f"\n✅ 이미지 저장: {args.save_images}")
            print(f"   총 {success_count}장")


if __name__ == '__main__':
    main()
