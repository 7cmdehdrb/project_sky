from omni.isaac.core.world import World
import time

# Isaac Sim 실행 중인지 확인 후 실행
world = World()

for _ in range(10):

    time.sleep(1)  # 1초 대기
    world.reset()

    print(world)  # 정상적으로 생성되었는지 확인
