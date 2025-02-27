import pyautogui
import time

try:
    while True:
        pyautogui.click()  # 마우스 좌클릭
        time.sleep(2)  # 2초 대기
except KeyboardInterrupt:
    print("프로그램 종료")
