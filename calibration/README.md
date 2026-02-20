# 캘리브레이션 파일

이 디렉토리에는 카메라 캘리브레이션 파일이 저장됩니다.

## 필요한 파일

### 1. charuco_calibration.yml
- **설명:** ChArUco 보드로 계산한 내부 파라미터
- **포함 내용:** 
  - 각 카메라의 내부 행렬 (camera_N_matrix)
  - 왜곡 계수 (camera_N_distortion)
  - N = 0~7 (카메라 1~8)

### 2. extrinsics_circular_45deg.yml
- **설명:** 45도 간격 원형 배치 외부 파라미터
- **포함 내용:**
  - 각 카메라의 회전 행렬 (R)
  - 이동 벡터 (t)
  - camera_1 ~ camera_8

## 파일 위치

기존 캘리브레이션 파일을 이 디렉토리에 복사하세요:

```bash
# 예제
cp /path/to/charuco_calibration.yml calibration/
cp /path/to/extrinsics_circular_45deg.yml calibration/
```

## 캘리브레이션 방법

자세한 캘리브레이션 방법은 메인 README.md를 참조하세요.
