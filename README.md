# pet_project

# 프로젝트 이름

이 프로젝트는 GCP Compute Engine에서 실행되며, e2-standard-2 인스턴스를 사용합니다.

## 사용법

아래의 단계를 따라 프로젝트를 실행할 수 있습니다.

1. 프로젝트 클론:

   ```bash
   git clone https://github.com/ta1231/pet_project.git
   cd pet_project

2. Python 가상 환경 설치:
   ```bash
   apt install python3.8-venv
3. 가상 환경 활성화:
    ```bash
   source venv/bin/activate
4. 필요한 패키지 설치:
    ```bash
   pip install tensorflow numpy pandas uvicorn fastapi scikit-learn
5. 어플리케이션 실행
    ```bash
    nohup uvicorn main:app --host 0.0.0.0 --port 8000 &
