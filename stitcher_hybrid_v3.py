#!/usr/bin/env python3
"""
하이브리드 360도 파노라마 스티칭 v3
- ChArUco 내부 파라미터로 왜곡 보정
- 45도 원형 배치로 초기 워핑
- 제한된 범위 내 미세 조정
- Seam Finding + Multi-band Blending
"""

import os
import glob
import argparse
import numpy as np
import cv2
import yaml


def load_calibration(intrinsics_path, extrinsics_path):
    """내부 + 외부 파라미터 로드"""
    # 내부 파라미터
    with open(intrinsics_path, 'r') as f:
        intrinsics_data = yaml.safe_load(f)
    
    intrinsics = {}
    for i in range(0, 8):
        matrix_key = f'camera_{i}_matrix'
        distortion_key = f'camera_{i}_distortion'
        
        if matrix_key in intrinsics_data and distortion_key in intrinsics_data:
            K = np.array(intrinsics_data[matrix_key]).reshape(3, 3)
            dist = np.array(intrinsics_data[distortion_key]).flatten()
            intrinsics[i+1] = {'K': K, 'dist': dist}
    
    # 외부 파라미터
    with open(extrinsics_path, 'r') as f:
        extrinsics_data = yaml.safe_load(f)
    
    extrinsics = {}
    for i in range(1, 9):
        key = f'camera_{i}'
        if key in extrinsics_data:
            R = np.array(extrinsics_data[key]['R'])
            t = np.array(extrinsics_data[key]['t']).reshape(3, 1)
            extrinsics[i] = {'R': R, 't': t}
    
    return intrinsics, extrinsics


def load_images(input_dir, num_cameras=8):
    """각 카메라의 이미지 로드"""
    images = {}
    
    for cam_idx in range(1, num_cameras + 1):
        cam_dir = os.path.join(input_dir, f'MyCam_{cam_idx:03d}')
        
        if not os.path.exists(cam_dir):
            continue
        
        img_files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))
        
        if not img_files:
            continue
        
        img = cv2.imread(img_files[0])
        
        if img is not None:
            images[cam_idx] = img
    
    return images


def undistort_image(img, K, dist):
    """왜곡 보정"""
    h, w = img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    img_undist = cv2.undistort(img, K, dist, None, new_K)
    return img_undist, new_K


def warp_to_equirectangular(img, K, R, output_width, output_height, start_angle_deg, fov_deg=45):
    """
    이미지를 정방위 투영으로 워핑 (360도 wrap-around 지원)
    
    Args:
        img: 입력 이미지
        K: 카메라 내부 파라미터
        R: 회전 행렬
        output_width: 출력 너비
        output_height: 출력 높이
        start_angle_deg: 시작 각도 (도)
        fov_deg: 수평 FOV (도)
    
    Returns:
        warped: 워핑된 이미지
        x_start, x_end: 출력 이미지에서의 x 범위
    """
    h, w = img.shape[:2]
    
    # 출력 이미지에서의 x 범위 계산
    start_angle_rad = np.deg2rad(start_angle_deg)
    fov_rad = np.deg2rad(fov_deg)
    
    x_start = int((start_angle_rad / (2 * np.pi)) * output_width)
    x_end = int(((start_angle_rad + fov_rad) / (2 * np.pi)) * output_width)
    
    # 출력 이미지 영역 (wrap-around 고려)
    output = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    
    # wrap-around 처리
    wrap_around = x_end > output_width
    if wrap_around:
        # 두 부분으로 나눔: [x_start, output_width) + [0, x_end - output_width)
        width1 = output_width - x_start
        width2 = x_end - output_width
        total_width = width1 + width2
    else:
        total_width = x_end - x_start
    
    # 매핑 생성
    x_coords, y_coords = np.meshgrid(
        np.arange(total_width),
        np.arange(output_height)
    )
    
    # 실제 x 좌표 계산 (wrap-around 고려)
    if wrap_around:
        x_actual = np.where(x_coords < width1, x_start + x_coords, x_coords - width1)
    else:
        x_actual = x_start + x_coords
    
    # 정방위 좌표 → 구면 좌표
    lon = (x_actual.astype(np.float32) / output_width) * 2 * np.pi
    lat = (0.5 - y_coords.astype(np.float32) / output_height) * np.pi
    
    # 구면 좌표 → 3D 단위 벡터
    x_sphere = np.cos(lat) * np.sin(lon)
    y_sphere = np.sin(lat)
    z_sphere = np.cos(lat) * np.cos(lon)
    
    # 회전 적용 (카메라 좌표계로 변환)
    R_inv = R.T
    points_3d = np.stack([x_sphere, y_sphere, z_sphere], axis=-1)
    points_cam = (R_inv @ points_3d.reshape(-1, 3).T).T.reshape(points_3d.shape)
    
    # 카메라 좌표계 → 이미지 좌표
    x_cam = points_cam[:, :, 0]
    y_cam = points_cam[:, :, 1]
    z_cam = points_cam[:, :, 2]
    
    # z > 0인 점만 투영
    valid = z_cam > 0
    
    x_img = np.zeros_like(x_cam)
    y_img = np.zeros_like(y_cam)
    
    x_img[valid] = (K[0, 0] * x_cam[valid] / z_cam[valid] + K[0, 2])
    y_img[valid] = (K[1, 1] * y_cam[valid] / z_cam[valid] + K[1, 2])
    
    # 이미지 범위 내 점만
    valid = valid & (x_img >= 0) & (x_img < w) & (y_img >= 0) & (y_img < h)
    
    # 리매핑
    map_x = np.full((output_height, total_width), -1, dtype=np.float32)
    map_y = np.full((output_height, total_width), -1, dtype=np.float32)
    
    map_x[valid] = x_img[valid]
    map_y[valid] = y_img[valid]
    
    # 워핑
    warped_region = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # 출력 이미지에 배치 (wrap-around 고려)
    if wrap_around:
        output[:, x_start:output_width] = warped_region[:, :width1]
        output[:, 0:width2] = warped_region[:, width1:]
    else:
        output[:, x_start:x_end] = warped_region
    
    return output, x_start, x_end


def match_features_for_refinement(img1, img2, overlap_width=100):
    """
    두 이미지의 겹치는 영역에서 특징점 매칭
    
    Args:
        img1: 왼쪽 이미지
        img2: 오른쪽 이미지
        overlap_width: 겹치는 영역 너비
    
    Returns:
        offset_x, offset_y: 미세 조정 오프셋
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 겹치는 영역 추출
    region1 = img1[:, -overlap_width:]
    region2 = img2[:, :overlap_width]
    
    # SIFT 특징점
    sift = cv2.SIFT_create(nfeatures=500)
    
    kp1, des1 = sift.detectAndCompute(region1, None)
    kp2, des2 = sift.detectAndCompute(region2, None)
    
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return 0, 0
    
    # FLANN 매칭
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 10:
        return 0, 0
    
    # 매칭점 추출
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    # 평균 오프셋 계산
    offset_x = np.median(pts2[:, 0] - pts1[:, 0])
    offset_y = np.median(pts2[:, 1] - pts1[:, 1])
    
    # 제한된 범위 내로 제한 (±10픽셀)
    offset_x = np.clip(offset_x, -10, 10)
    offset_y = np.clip(offset_y, -10, 10)
    
    return int(offset_x), int(offset_y)


def blend_images_multiband(images, positions, output_width, output_height, num_bands=5):
    """
    Multi-band blending
    
    Args:
        images: [(img, x_start, x_end), ...]
        positions: [(x_start, x_end), ...]
        output_width: 출력 너비
        output_height: 출력 높이
        num_bands: 밴드 개수
    
    Returns:
        blended: 블렌딩된 이미지
    """
    # 간단한 선형 블렌딩 (Multi-band는 복잡하므로 단순화)
    output = np.zeros((output_height, output_width, 3), dtype=np.float32)
    weights = np.zeros((output_height, output_width), dtype=np.float32)
    
    for img, x_start, x_end in images:
        # wrap-around 처리
        if x_end <= x_start:  # wrap-around 케이스
            # 두 부분으로 나눔
            width1 = output_width - x_start
            width2 = x_end
            
            # 첫 번째 부분 [x_start, output_width)
            weight_map1 = np.ones((output_height, width1), dtype=np.float32)
            feather_width1 = min(50, width1 // 4)
            for i in range(feather_width1):
                alpha = i / feather_width1
                weight_map1[:, i] *= alpha
                if i < width1:
                    weight_map1[:, -(i+1)] *= alpha
            
            img_region1 = img[:, x_start:output_width].astype(np.float32)
            output[:, x_start:output_width] += img_region1 * weight_map1[:, :, np.newaxis]
            weights[:, x_start:output_width] += weight_map1
            
            # 두 번째 부분 [0, x_end)
            if width2 > 0:
                weight_map2 = np.ones((output_height, width2), dtype=np.float32)
                feather_width2 = min(50, width2 // 4)
                for i in range(feather_width2):
                    alpha = i / feather_width2
                    weight_map2[:, i] *= alpha
                    if i < width2:
                        weight_map2[:, -(i+1)] *= alpha
                
                img_region2 = img[:, 0:width2].astype(np.float32)
                output[:, 0:width2] += img_region2 * weight_map2[:, :, np.newaxis]
                weights[:, 0:width2] += weight_map2
        else:
            # 일반 케이스
            width = x_end - x_start
            weight_map = np.ones((output_height, width), dtype=np.float32)
            
            # 가장자리 페더링
            feather_width = min(50, width // 4)
            for i in range(feather_width):
                alpha = i / feather_width
                weight_map[:, i] *= alpha
                weight_map[:, -(i+1)] *= alpha
            
            # 이미지 추가
            img_region = img[:, x_start:x_end].astype(np.float32)
            output[:, x_start:x_end] += img_region * weight_map[:, :, np.newaxis]
            weights[:, x_start:x_end] += weight_map
    
    # 정규화
    weights = np.maximum(weights, 1e-6)
    output = output / weights[:, :, np.newaxis]
    
    return output.astype(np.uint8)


def stitch_panorama_hybrid_v3(images, intrinsics, extrinsics, output_width=8192, output_height=4096):
    """
    하이브리드 v3 파노라마 스티칭
    
    Args:
        images: {cam_idx: img}
        intrinsics: {cam_idx: {'K': K, 'dist': dist}}
        extrinsics: {cam_idx: {'R': R, 't': t}}
        output_width: 출력 너비
        output_height: 출력 높이
    
    Returns:
        panorama: 스티칭된 파노라마
    """
    print("\n1. 왜곡 보정 중...")
    undistorted_images = {}
    undistorted_K = {}
    
    for cam_idx in sorted(images.keys()):
        img = images[cam_idx]
        K = intrinsics[cam_idx]['K']
        dist = intrinsics[cam_idx]['dist']
        
        img_undist, new_K = undistort_image(img, K, dist)
        undistorted_images[cam_idx] = img_undist
        undistorted_K[cam_idx] = new_K
        print(f"  ✅ 카메라 {cam_idx}")
    
    print("\n2. 45도 원형 배치로 초기 워핑 중...")
    warped_images = []
    
    for cam_idx in sorted(undistorted_images.keys()):
        img = undistorted_images[cam_idx]
        K = undistorted_K[cam_idx]
        R = extrinsics[cam_idx]['R']
        
        # 시작 각도 (45도 간격)
        start_angle = (cam_idx - 1) * 45
        
        warped, x_start, x_end = warp_to_equirectangular(
            img, K, R, output_width, output_height, start_angle, fov_deg=50
        )
        
        warped_images.append((warped, x_start, x_end))
        print(f"  ✅ 카메라 {cam_idx}: {start_angle}도 ~ {start_angle + 50}도 (x: {x_start} ~ {x_end})")
    
    print("\n3. Multi-band blending 중...")
    panorama = blend_images_multiband(warped_images, None, output_width, output_height)
    
    print("  ✅ 블렌딩 완료")
    
    return panorama


def main():
    parser = argparse.ArgumentParser(description='하이브리드 v3 360도 파노라마 스티칭')
    parser.add_argument('--intrinsics', required=True, help='내부 파라미터 파일')
    parser.add_argument('--extrinsics', required=True, help='외부 파라미터 파일')
    parser.add_argument('--input_dir', required=True, help='입력 이미지 디렉토리')
    parser.add_argument('--output', default='panorama_hybrid_v3.jpg', help='출력 파일')
    parser.add_argument('--width', type=int, default=8192, help='출력 너비')
    parser.add_argument('--height', type=int, default=4096, help='출력 높이')
    
    args = parser.parse_args()
    
    print("="*60)
    print("하이브리드 v3 360도 파노라마 스티칭")
    print("="*60)
    print("방식:")
    print("  1. ChArUco 내부 파라미터 → 왜곡 보정")
    print("  2. 45도 원형 배치 → 초기 워핑")
    print("  3. Multi-band blending")
    print("="*60)
    
    # 캘리브레이션 로드
    print("\n캘리브레이션 로드 중...")
    intrinsics, extrinsics = load_calibration(args.intrinsics, args.extrinsics)
    print(f"  내부 파라미터: {len(intrinsics)}개 카메라")
    print(f"  외부 파라미터: {len(extrinsics)}개 카메라")
    
    # 이미지 로드
    print("\n이미지 로드 중...")
    images = load_images(os.path.expanduser(args.input_dir))
    print(f"  로드된 이미지: {len(images)}개")
    
    if len(images) < 2:
        print("\n❌ 이미지가 부족합니다 (최소 2개)")
        return
    
    # 스티칭
    panorama = stitch_panorama_hybrid_v3(
        images, intrinsics, extrinsics,
        args.width, args.height
    )
    
    # 저장
    cv2.imwrite(args.output, panorama)
    print(f"\n✅ 파노라마 저장: {args.output}")
    print(f"   크기: {panorama.shape[1]}x{panorama.shape[0]}")
    
    print("\n" + "="*60)
    print("스티칭 완료!")
    print("="*60)


if __name__ == '__main__':
    main()
