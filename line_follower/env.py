import math 

from .simulator import Simulator 

class LineFollowerEnv: 
    "기존 시뮬레이터를 학습용으로 연결할 클래스"

    def __init__(self): 
        self .sim = Simulator()
    
    def reset(self):
        # 로봇을 초기 위치로 돌리고 센서값을 반환한다.
        return self.sim.reset() 
    def step(self,action):
        #행동을 기존 시뮬레이터에 전달하고 센서값을 반환한다. 
        return self.sim.step(action)
    def distance_error(self):
        #원의 중심에서 로봇까지의 거리 
        distance = math.hypot(self.sim.robot.x,self.sim.robot.y)

        #원형 선의 반지름을 빼면 선 중심에서의 오차 
        return distance - self.sim.track.radius 
