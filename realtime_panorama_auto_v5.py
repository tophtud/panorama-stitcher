#!/usr/bin/env python3
"""
8대 카메라 자동 스티칭 v5 (카메라 순서 재배치)
- 카메라 순서 지정 가능
- 구면 투영
- 노출 보정
- 강화된 블렌딩
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
    
    camera_images = {}
    
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
            camera_images[cam_idx] = images
            print(f"  ✅ 카메라 {cam_idx}: {len(images)}장")
    
    print(f"\n✅ 총 {len(camera_images)}개 카메라")
    
    # 순서대로 정렬
    ordered_images = {}
    for i, cam_idx in enumerate(camera_order):
        if cam_idx in camera_images:
            ordered_images[i+1] = camera_images[cam_idx]
    
    return ordered_images, camera_order


class AutoStitcherV5:
    """
    자동 스티칭 v5 (카메라 순서 재배치)
    """
    
    def __init__(self, scale=0.5, brightness_boost=1.2, blend_strength=4.0):
        self.scale = scale
        self.brightness_boost = brightness_boost
        self.blend_strength = blend_strength
        self.cameras = None
        self.avg_focal = None
        self.xmaps = []
        self.ymaps = []
        self.corners = []
        self.sizes = []
        self.masks = []
        self.pano_size = None
        self.pano_offset = None
        self.input_size = None
        self.is_initialized = False
        self.exposure_gains = []
    
    def calibrate(self, images):
        """
        OpenCV Stitcher로 자동 캘리브레이션
        """
        print("\n" + "="*60)
        print(f"자동 스티칭 v5 (카메라 순서 재배치)")
        print(f"스케일: {int(self.scale*100)}%")
        print(f"밝기 부스트: {self.brightness_boost}x")
        print(f"블렌딩 강도: {self.blend_strength}")
        print("="*60)
        
        # 리사이즈
        images_scaled = [
            cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            for img in images
        ]
        
        h, w = images_scaled[0].shape[:2]
        self.input_size = (w, h)
        
        print(f"  입력: {len(images_scaled)}개 카메라")
        print(f"  크기: {w} x {h}")
        
        # 노출 보정 계산
        print("\n  1. 노출 보정 계산 중...")
        self._calculate_exposure_compensation(images_scaled)
        
        # 노출 보정 적용
        images_compensated = []
        for img, gain in zip(images_scaled, self.exposure_gains):
            compensated = cv2.convertScaleAbs(img, alpha=gain, beta=0)
            images_compensated.append(compensated)
        
        # OpenCV Stitcher
        print("\n  2. OpenCV Stitcher 실행 중...")
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        
        status = stitcher.estimateTransform(images_compensated)
        
        if status != cv2.Stitcher_OK:
            print(f"  ❌ 캘리브레이션 실패: {status}")
            return None
        
        print(f"  ✅ 변환 추정 성공!")
        
        # 카메라 파라미터 추출
        self.cameras = stitcher.cameras()
        self.avg_focal = np.mean([cam.focal for cam in self.cameras])
        
        print(f"  카메라 수: {len(self.cameras)}")
        print(f"  평균 focal: {self.avg_focal:.1f}")
        
        # 기준 파노라마 생성
        print("\n  3. 기준 파노라마 생성 중...")
        status2, reference = stitcher.composePanorama()
        
        if status2 != cv2.Stitcher_OK:
            print(f"  ❌ 파노라마 생성 실패: {status2}")
            return None
        
        # 밝기 부스트 적용
        reference = self._apply_brightness_boost(reference)
        
        print(f"  ✅ 기준 파노라마 생성 성공!")
        print(f"  크기: {reference.shape[1]} x {reference.shape[0]}")
        
        # 워핑 맵 생성
        print("\n  4. 구면 워핑 맵 생성 중...")
        self._build_spherical_warp_maps()
        
        self.is_initialized = True
        
        print("\n✅ 캘리브레이션 완료!")
        print("="*60)
        
        return reference
    
    def _calculate_exposure_compensation(self, images):
        """
        노출 보정 계산
        """
        brightnesses = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            brightnesses.append(brightness)
        
        target_brightness = np.median(brightnesses)
        
        self.exposure_gains = []
        for brightness in brightnesses:
            gain = target_brightness / (brightness + 1e-6)
            gain = np.clip(gain, 0.8, 1.2)
            self.exposure_gains.append(gain)
        
        print(f"    노출 게인: {[f'{g:.2f}' for g in self.exposure_gains]}")
    
    def _apply_brightness_boost(self, image):
        """
        밝기 부스트 적용
        """
        if self.brightness_boost == 1.0:
            return image
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * self.brightness_boost, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return result
    
    def _build_spherical_warp_maps(self):
        """
        구면 투영 워핑 맵 생성
        """
        w, h = self.input_size
        
        warper = cv2.PyRotationWarper('spherical', self.avg_focal)
        
        K = np.array([
            [self.avg_focal, 0, w/2],
            [0, self.avg_focal, h/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.xmaps = []
        self.ymaps = []
        self.corners = []
        self.sizes = []
        self.masks = []
        
        for i, cam in enumerate(self.cameras):
            R = cam.R.astype(np.float32)
            
            roi, xmap, ymap = warper.buildMaps((w, h), K, R)
            
            self.corners.append((roi[0], roi[1]))
            self.sizes.append((roi[2], roi[3]))
            self.xmaps.append(xmap)
            self.ymaps.append(ymap)
            
            mask = ((xmap >= 0) & (xmap < w) & (ymap >= 0) & (ymap < h)).astype(np.float32)
            self.masks.append(mask)
        
        # 파노라마 크기 계산
        min_x = min(c[0] for c in self.corners)
        min_y = min(c[1] for c in self.corners)
        max_x = max(c[0] + s[0] for c, s in zip(self.corners, self.sizes))
        max_y = max(c[1] + s[1] for c, s in zip(self.corners, self.sizes))
        
        self.pano_offset = (min_x, min_y)
        self.pano_size = (max_x - min_x, max_y - min_y)
        
        print(f"    파노라마 크기: {self.pano_size[0]} x {self.pano_size[1]}")
    
    def stitch(self, images):
        """
        이미지 스티칭
        """
        if not self.is_initialized:
            return None
        
        # 리사이즈 + 노출 보정
        images_scaled = []
        for img, gain in zip(images, self.exposure_gains):
            scaled = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            compensated = cv2.convertScaleAbs(scaled, alpha=gain, beta=0)
            images_scaled.append(compensated)
        
        pano_w, pano_h = self.pano_size
        offset_x, offset_y = self.pano_offset
        
        result = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
        weight_sum = np.zeros((pano_h, pano_w), dtype=np.float32)
        
        w, h = self.input_size
        
        for img, xmap, ymap, mask, corner in zip(
            images_scaled, self.xmaps, self.ymaps, self.masks, self.corners
        ):
            # 이중선형 보간 워핑
            img_float = img.astype(np.float32)
            
            x0 = np.floor(xmap).astype(np.int32)
            y0 = np.floor(ymap).astype(np.int32)
            x1 = np.clip(x0 + 1, 0, w - 1)
            y1 = np.clip(y0 + 1, 0, h - 1)
            x0 = np.clip(x0, 0, w - 1)
            y0 = np.clip(y0, 0, h - 1)
            
            wx = (xmap - np.floor(xmap))[:, :, np.newaxis]
            wy = (ymap - np.floor(ymap))[:, :, np.newaxis]
            
            warped = (
                img_float[y0, x0] * (1 - wx) * (1 - wy) +
                img_float[y0, x1] * wx * (1 - wy) +
                img_float[y1, x0] * (1 - wx) * wy +
                img_float[y1, x1] * wx * wy
            )
            
            # 목적지 좌표
            dst_x = corner[0] - offset_x
            dst_y = corner[1] - offset_y
            
            wh, ww = warped.shape[:2]
            
            # 경계 처리
            src_x1, src_y1, src_x2, src_y2 = 0, 0, ww, wh
            
            if dst_x < 0:
                src_x1 = -dst_x
                dst_x = 0
            if dst_y < 0:
                src_y1 = -dst_y
                dst_y = 0
            if dst_x + (src_x2 - src_x1) > pano_w:
                src_x2 = src_x1 + (pano_w - dst_x)
            if dst_y + (src_y2 - src_y1) > pano_h:
                src_y2 = src_y1 + (pano_h - dst_y)
            
            if src_x2 <= src_x1 or src_y2 <= src_y1:
                continue
            
            # 강화된 가우시안 가중치
            center_x = (src_x2 + src_x1) / 2
            center_y = (src_y2 + src_y1) / 2
            
            yy, xx = np.mgrid[src_y1:src_y2, src_x1:src_x2]
            
            dist = np.sqrt(
                ((xx - center_x) / (ww / 2)) ** 2 +
                ((yy - center_y) / (wh / 2)) ** 2
            )
            
            blend = np.exp(-self.blend_strength * dist ** 2) * mask[src_y1:src_y2, src_x1:src_x2]
            
            # 누적
            result[dst_y:dst_y+(src_y2-src_y1), dst_x:dst_x+(src_x2-src_x1)] += \
                warped[src_y1:src_y2, src_x1:src_x2] * blend[:, :, np.newaxis]
            
            weight_sum[dst_y:dst_y+(src_y2-src_y1), dst_x:dst_x+(src_x2-src_x1)] += blend
        
        # 정규화
        panorama = result / np.maximum(weight_sum[:, :, np.newaxis], 1e-6)
        panorama = np.clip(panorama, 0, 255).astype(np.uint8)
        
        # 밝기 부스트 적용
        panorama = self._apply_brightness_boost(panorama)
        
        return panorama


def main():
    parser = argparse.ArgumentParser(
        description='8대 카메라 자동 스티칭 v5 (카메라 순서 재배치)'
    )
    parser.add_argument('--mode', choices=['test', 'realtime'], default='test')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--num_frames', type=int, default=10)
    parser.add_argument('--reference_frame', type=int, default=0)
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--brightness_boost', type=float, default=1.2)
    parser.add_argument('--blend_strength', type=float, default=4.0)
    parser.add_argument('--camera_order', type=int, nargs='+', help='카메라 순서 (예: 5 4 3 2 1 8 7 6)')
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
    print("8대 카메라 자동 스티칭 v5 (카메라 순서 재배치)")
    print("="*60)
    print(f"모드: {args.mode}")
    print(f"기준 프레임: {args.reference_frame}")
    print(f"스케일: {args.scale * 100:.0f}%")
    print(f"밝기 부스트: {args.brightness_boost}x")
    print(f"블렌딩 강도: {args.blend_strength}")
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
    camera_images, camera_order = load_camera_images(
        os.path.expanduser(args.input_dir),
        num_frames=args.num_frames,
        crop_edges=crop_config,
        camera_order=args.camera_order
    )
    
    if len(camera_images) < 2:
        print("\n❌ 카메라가 부족합니다")
        return
    
    cam_indices = sorted(camera_images.keys())
    num_frames = len(camera_images[cam_indices[0]])
    
    print(f"\n실제 사용 카메라: {[camera_order[i-1] for i in cam_indices]}")
    print(f"프레임 수: {num_frames}")
    
    # 스티처 초기화
    stitcher = AutoStitcherV5(
        scale=args.scale,
        brightness_boost=args.brightness_boost,
        blend_strength=args.blend_strength
    )
    
    ref_images = []
    for cam_idx in cam_indices:
        ref_idx = min(args.reference_frame, len(camera_images[cam_idx]) - 1)
        ref_images.append(camera_images[cam_idx][ref_idx])
    
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
            
            images = []
            for cam_idx in cam_indices:
                idx = frame_idx % len(camera_images[cam_idx])
                images.append(camera_images[cam_idx][idx])
            
            panorama = stitcher.stitch(images)
            
            if panorama is not None:
                success_count += 1
                
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                fps_list.append(fps)
                
                avg_fps = np.mean(fps_list[-30:])
                cv2.putText(
                    panorama,
                    f"FPS: {avg_fps:.1f} | Frame: {frame_idx+1}/{num_frames} | Auto v5",
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
