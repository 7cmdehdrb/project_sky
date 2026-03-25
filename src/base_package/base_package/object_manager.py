import re
from collections import UserDict
import hashlib


"""
manager = ObjectManager(node=None)

# 완벽히 일치하는 경우
print(manager.names["can_1"])  
# 출력: 'coca_cola'

print(manager.names["mug_2"])  
# 출력: 'mug_gray'

# 1) 앞뒤로 언더스코어(_)가 붙은 잡음이 섞인 경우
print(manager.names["cup_3_blue"])  
# 출력: 'cup_blue' (키인 "cup_3" 또는 값인 "cup_blue"를 인식)

# 2) 순서가 뒤바뀌고 이상한 단어가 붙은 경우
print(manager.names["my_cup_blue_3"])  
# 출력: 'cup_blue' ("cup_blue"를 인식)

# 3) 값(value) 자체의 일부가 들어온 경우
print(manager.names["detect_mug_black_obj"]) 
# 출력: 'mug_black'
"""


class SmartNameDict(UserDict):
    """
    정규식을 활용해 유연한 문자열 매칭을 지원하는 커스텀 딕셔너리 클래스입니다.
    기존 딕셔너리처럼 동작하지만, 매칭 실패 시 내부적으로 정규식을 돌려 값을 찾아냅니다.
    """

    def __init__(self, initial_mapping=None):
        super().__init__()
        self._patterns = {}
        if initial_mapping:
            for k, v in initial_mapping.items():
                self[k] = v  # __setitem__ 을 호출하여 정규식 자동 컴파일

    def __setitem__(self, key, value):
        # 1. 원본 데이터 저장
        self.data[key] = value

        # 2. 정규식 패턴 컴파일 및 저장 (딕셔너리가 확장될 때마다 자동 업데이트 됨)
        # 조건: 앞뒤가 문자열의 시작/끝(^, $)이거나 언더스코어(_)로 구분된 독립된 단어일 것
        # 예시: 'cup_3' 또는 'cup_blue'가 입력 문자열 안에 온전히 존재해야 매칭 (can_10 오탐 방지)
        pattern = rf"(?:^|_)(?:{key}|{value})(?:_|$)"
        self._patterns[value] = re.compile(pattern, re.IGNORECASE)

    def __getitem__(self, key):
        # 1. 완벽히 일치하는 키가 있으면 바로 반환 O(1)
        if key in self.data:
            return self.data[key]

        # 2. 일치하는 키가 없으면 정규식으로 유연한 탐색 진행
        for real_name, pattern in self._patterns.items():
            real_name: str
            pattern: re.Pattern

            if pattern.search(str(key)):
                return real_name

        # 3. 매칭되는 게 아예 없으면 원래 입력값 반환 (혹은 상황에 따라 KeyError 발생)
        return key


class ObjectManager:
    def __init__(self, *args, **kwargs):

        # "스마트 딕셔너리" 인스턴스화
        self.names = SmartNameDict(
            {
                "can_1": "coca_cola",
                "can_2": "cyder",
                "can_3": "yello_peach",
                "cup_1": "cup_sky",
                "cup_2": "cup_white",
                "cup_3": "cup_blue",
                "mug_1": "mug_black",
                "mug_2": "mug_gray",
                "mug_3": "mug_yello",
                "bottle_1": "alive",
                "bottle_2": "green_tea",
                "bottle_3": "yello_smoothie",
            }
        )

        self.names = SmartNameDict(
            {
                "can_1": "coca_cola",
                "can_2": "cyder",
                "can_3": "yello_peach",
                "can_4": "catata",
                "cup_1": "cup_sky",
                "cup_2": "cup_white",
                "cup_3": "cup_blue",
                "cup_4": "cup_green",
                "mug_1": "mug_black",
                "mug_2": "mug_gray",
                "mug_3": "mug_yello",
                "mug_4": "mug_orange",
                "bottle_1": "alive",
                "bottle_2": "green_tea",
                "bottle_3": "yello_smoothie",
                "bottle_4": "bottle_red",
            }
        )

        # 기타 인덱스 맵핑은 .data (순수 dict)를 기반으로 기존과 동일하게 생성
        self.classes = {v: k for k, v in self.names.data.items()}
        self.indexs = {k: i for i, k in enumerate(self.names.data.keys())}
        self.reverse_indexs = {i: k for i, k in enumerate(self.names.data.keys())}

    def get_color(self, text: str) -> tuple:
        """
        텍스트를 규격화된 이름으로 변환한 뒤, 해당 객체만의 고유한 RGB(BGR) 튜플을 반환합니다.
        """
        # 1. 입력 텍스트를 스마트하게 규격화된 이름으로 변환
        # 예: "detected_can_1" -> "coca_cola"
        normalized_name = self.names[text]

        # 2. 규격화된 이름을 바탕으로 MD5 해시 생성
        hash_object = hashlib.md5(normalized_name.encode("utf-8"))

        # 3. 16진수 해시값을 정수로 변환 후 비트 연산으로 RGB 추출
        hash_int = int(hash_object.hexdigest(), 16)
        r = (hash_int & 0xFF0000) >> 16
        g = (hash_int & 0x00FF00) >> 8
        b = hash_int & 0x0000FF

        # OpenCV 시각화에 바로 쓸 수 있도록 튜플로 반환
        return (r, g, b)

    def get_object_id(self, text: str) -> int:
        """
        입력된 문자열을 정규화된 이름으로 변환한 뒤, 해당 객체가 등록된 순서(ID)를 정수로 반환합니다.
        매칭되는 객체가 없을 경우 -1을 반환합니다.
        """
        # 1. 입력 텍스트 정상화 (예: "detected_can_1_obj" -> "coca_cola")
        normalized_value = self.names[text]

        # 2. 정규화된 값을 통해 원래의 Key 탐색 (예: "coca_cola" -> "can_1")
        # SmartNameDict 매칭 실패 시 입력값이 그대로 반환되므로, 에러 방지를 위해 .get() 사용
        original_key = self.classes.get(normalized_value)

        # 3. Key가 존재하면 해당 ID(인덱스) 반환, 없으면 -1 반환
        if original_key is not None:
            return self.indexs.get(original_key, -1)

        return -1
