# 8대 카메라 360도 파노라마 스티칭 시스템

8대 카메라로 촬영한 이미지를 실시간으로 360도 파노라마로 스티칭하는 시스템입니다.

## 🎯 주요 기능

- ✅ **ChArUco 캘리브레이션** - 정확한 내부 파라미터 계산
- ✅ **45도 원형 배치** - 고정된 카메라 배치로 안정적인 스티칭
- ✅ **왜곡 보정** - 렌즈 왜곡 자동 보정
- ✅ **실시간 처리** - 멀티스레드 프레임 캡처 및 스티칭
- ✅ **Multi-band Blending** - 자연스러운 이음새
- ✅ **테스트 모드** - 저장된 이미지로 성능 테스트

## 📁 파일 구조

```
panorama_stitcher/
├── stitcher_hybrid_v3.py      # 단일 이미지 스티칭
├── realtime_stitcher.py        # 실시간 스티칭 시스템
├── calibration/
│   ├── charuco_calibration.yml # 내부 파라미터
│   └── extrinsics_circular_45deg.yml # 외부 파라미터
├── examples/
│   └── stream_urls.yml         # 스트리밍 URL 설정 예제
└── README.md
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# Python 3.7 이상 필요
pip install opencv-python opencv-contrib-python numpy pyyaml

# 또는 가상환경 사용
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python opencv-contrib-python numpy pyyaml
```

### 2. 단일 이미지 스티칭

```bash
python3 stitcher_hybrid_v3.py \
    --intrinsics calibration/charuco_calibration.yml \
    --extrinsics calibration/extrinsics_circular_45deg.yml \
    --input_dir /path/to/images \
    --output panorama.jpg
```

**입력 디렉토리 구조:**
```
input_dir/
├── MyCam_001/
│   └── image_001.jpg
├── MyCam_002/
│   └── image_001.jpg
...
└── MyCam_008/
    └── image_001.jpg
```

### 3. 실시간 스티칭 (테스트 모드)

```bash
# 10장 테스트
python3 realtime_stitcher.py \
    --mode test \
    --intrinsics calibration/charuco_calibration.yml \
    --extrinsics calibration/extrinsics_circular_45deg.yml \
    --input_dir /path/to/images \
    --num_frames 10

# 비디오 저장
python3 realtime_stitcher.py \
    --mode test \
    --intrinsics calibration/charuco_calibration.yml \
    --extrinsics calibration/extrinsics_circular_45deg.yml \
    --input_dir /path/to/images \
    --num_frames 10 \
    --save_video output.mp4
```

### 4. 실시간 스티칭 (라즈베리파이 연동)

```bash
# 1. stream_urls.yml 작성
cat > stream_urls.yml << EOF
1: "rtsp://192.168.1.101:8554/stream"
2: "rtsp://192.168.1.102:8554/stream"
3: "rtsp://192.168.1.103:8554/stream"
4: "rtsp://192.168.1.104:8554/stream"
5: "rtsp://192.168.1.105:8554/stream"
6: "rtsp://192.168.1.106:8554/stream"
7: "rtsp://192.168.1.107:8554/stream"
8: "rtsp://192.168.1.108:8554/stream"
EOF

# 2. 실행
python3 realtime_stitcher.py \
    --mode realtime \
    --intrinsics calibration/charuco_calibration.yml \
    --extrinsics calibration/extrinsics_circular_45deg.yml \
    --streams stream_urls.yml \
    --save_video realtime_panorama.mp4
```

## 📊 성능

- **처리 속도:** 5-15 FPS (시스템 성능에 따라)
- **출력 해상도:** 8192x4096 (기본값, 조정 가능)
- **메모리 사용:** ~2-4 GB

## 🔧 고급 옵션

### 출력 해상도 조정

```bash
python3 realtime_stitcher.py \
    --mode test \
    --intrinsics calibration/charuco_calibration.yml \
    --extrinsics calibration/extrinsics_circular_45deg.yml \
    --input_dir /path/to/images \
    --output_width 4096 \
    --output_height 2048
```

### 화면 표시 없이 실행

```bash
python3 realtime_stitcher.py \
    --mode test \
    --intrinsics calibration/charuco_calibration.yml \
    --extrinsics calibration/extrinsics_circular_45deg.yml \
    --input_dir /path/to/images \
    --no_display \
    --save_video output.mp4
```

## 📱 라즈베리파이 스트리밍 설정

### RTSP 스트리밍 (권장)

```bash
# 라즈베리파이에서 실행
ffmpeg -f v4l2 -i /dev/video0 \
    -c:v libx264 \
    -preset ultrafast \
    -tune zerolatency \
    -b:v 2M \
    -r 30 \
    -s 1280x720 \
    -f rtsp rtsp://0.0.0.0:8554/stream
```

### HTTP 스트리밍

```bash
# mjpg-streamer 사용
mjpg_streamer -i "input_uvc.so -d /dev/video0 -r 1280x720 -f 30" \
    -o "output_http.so -p 8080"
```

## 🐛 문제 해결

### 1. 캘리브레이션 파일 없음

```bash
# 캘리브레이션 파일 경로 확인
ls calibration/charuco_calibration.yml
ls calibration/extrinsics_circular_45deg.yml

# 전체 경로 사용
python3 stitcher_hybrid_v3.py \
    --intrinsics /full/path/to/charuco_calibration.yml \
    --extrinsics /full/path/to/extrinsics_circular_45deg.yml \
    --input_dir /path/to/images
```

### 2. 이미지 로드 실패

```bash
# 디렉토리 구조 확인
ls -R /path/to/images

# 각 카메라 폴더에 이미지가 있는지 확인
# MyCam_001, MyCam_002, ..., MyCam_008
```

### 3. 스트리밍 연결 실패

```bash
# 네트워크 확인
ping 192.168.1.101

# 포트 확인
telnet 192.168.1.101 8554

# VLC로 테스트
vlc rtsp://192.168.1.101:8554/stream
```

### 4. FPS 낮음

- 출력 해상도 낮추기 (`--output_width 4096 --output_height 2048`)
- 입력 해상도 낮추기 (라즈베리파이에서 1280x720 사용)
- 프레임 수 줄이기 (`--num_frames 5`)

## 📖 알고리즘 설명

### 하이브리드 v3 스티칭 파이프라인

1. **왜곡 보정**
   - ChArUco 캘리브레이션으로 얻은 내부 파라미터 사용
   - 렌즈 왜곡 자동 보정

2. **45도 원형 배치**
   - 8개 카메라를 45도 간격으로 고정 배치
   - 각 카메라는 50도 FOV로 워핑

3. **정방위 투영**
   - 구면 좌표계로 변환
   - 360도 wrap-around 처리

4. **Multi-band Blending**
   - 가장자리 페더링으로 자연스러운 이음새
   - 가중치 기반 블렌딩

## 🤝 기여

버그 리포트 및 기능 제안은 이슈로 등록해 주세요.

## 📄 라이선스

MIT License

## 👥 개발자

- 뉴딕스 팀

## 📞 문의

- 이슈 트래커: GitHub Issues
- 이메일: contact@example.com

```bash
python3 realtime_panorama_stitcher_v6.py \
    --mode test \
    --input_dir ~/뉴딕스\ 작업파일/20251228panoram_test_v2/calibration_data_chain_edge_8cam \
    --num_frames 10 \
    --reference_frame 7 \
    --scale 0.8 \
    --camera_order 5 4 3 2 1 8 7 6 \
    --try_use_gpu \
    --crop_edges 2 \
    --save_images panorama_v6_gpu \
    --save_reference reference_v6_gpu.jpg
```
