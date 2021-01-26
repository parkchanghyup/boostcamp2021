
# 부스트 캠프 3일차
---

## 파이썬 기본 자료구조



### 스택
---
- Last In First Out
- Data의 입력을 Push, 출력을 Pop 라고함


```python
a = [1,2,3,4,5]
a.append(10)   # push
print(a)
a.pop()        # pop
print(a)
```

    [1, 2, 3, 4, 5, 10]
    [1, 2, 3, 4, 5]
    

### 큐
---
- First In First Out
- 스택과 반대


```python
a = [1,2,3,4,5]
a.append(10)
print(a)
a.pop(0)
print(a)
```

    [1, 2, 3, 4, 5, 10]
    [2, 3, 4, 5, 10]
    

### 튜플
---
- 값의 변경이 불가능한 리스트
- 선언시 '[]' 가 아닌 "()"를 사용


```python
a=  [1,2,3,4] # 리스트
b = (1,2,3,4)
print(type(a),type(b))
```

    <class 'list'> <class 'tuple'>
    


```python
a[0] = 0 # 변경 가능(리스트)
b[0] = 0 # 변경 불가(튜플)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-4-9d4ebbbeda57> in <module>
          1 a[0] = 0 # 변경 가능(리스트)
    ----> 2 b[0] = 0 # 변경 불가(튜플)
    

    TypeError: 'tuple' object does not support item assignment



```python
c = (1)  # int로 인식
d = (1,) # tuple로 인식
print(type(c),type(d))
```

### set
---
- 값을 순서없이 저장, 중복 불허 하는 자료형


```python
s = set([1,2,3,1,2,3])
print(s)
```


```python
s.update([1,4,5,6,7])
print(s)

s.discard(3)
print(s)
```

- 수학에서 활용하는 다양한 집합 연산 가능


```python
s1 = set([1,2,3,4,5])
s2 = set([3,4,5,6,7])
s1.union(s2)
```


```python
s1 |s2
```


```python
 # 교집합
s1.intersection(s2)
s1 & s2
```


```python
 # s1과 s2의 차집합
s1.difference(s2)
s1 - s2
```

### dict
---
- Key 값을 활용하여, 데이터 값(Value)를 관리함


```python
country_code = {} # Dict 생성, country_code = dict() 도 가능
country_code = {'America': 1, 'Korea': 82, 'China': 86, 'Japan': 81}
country_code
```


```python
country_code.items() # Dict 데이터 출력
```


```python
country_code.keys() # Dict 키 값만 출력
```


```python
country_code["German"]= 49 # Dict 추가
country_code
```


```python
country_code.values() # Dict Value만 출력

```


```python
import csv
def getKey(item): # 정렬을 위한 함수
    return item[1] # 신경 쓸 필요 없음


command_data = [] # 파일 읽어오기

with open("command_data.csv", "r",encoding='utf-8') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in spamreader:
        command_data.append(row)
        
        
command_counter = {} # dict 생성, 아이디를 key값, 입력줄수를 value값
for data in command_data: # list 데이터를 dict로 변경
    if data[1] in command_counter.keys(): # 아이디가 이미 Key값으로 변경되었을 때
        command_counter[data[1]] += 1 # 기존 출현한 아이디
    else:
        command_counter[data[1]] = 1 # 처음 나온 아이디
        
dictlist = [] # dict를 list로 변경
for key, value in command_counter.items():
    temp = [key,value]
    dictlist.append(temp)
    
sorted_dict= sorted(dictlist, key=getKey, reverse=True) # list를 입력 줄 수로 정렬
print (sorted_dict[:100]) 
```

## collections
---
- List, Tuple, Dict에 대한 Python Built-in 확장 자료 구조(모듈)
- 편의성, 실행 효율 등을 사용자에게 제공함
- 아래의 모듈이 존재함
![image.png](attachment:image.png)

### deque
---
- Stack과 Queue를 지원하는 모듈
- List에 비해 효율적 (빠름)
- Lotate, reverse등 Linked List의 특성을 지원함


```python
from collections import deque
deque_list = deque()
for i in range(5):
    deque_list.append(i)
print(deque_list)
deque_list.appendleft(10)
print(deque_list)
```


```python
deque_list.rotate(1) # 시계 방향
print(deque_list)
print(deque(reversed(deque_list))) # 역순
```


```python
deque_list.extend([5, 6, 7])
print(deque_list)
deque_list.extendleft([5, 6, 7])
print(deque_list)
```

### 덱 vs List 시간비교 


```python
from collections import deque
import time
start_time = time.clock()
deque_list = deque()
# Stack
for i in range(10000):
    for i in range(10000):
        deque_list.append(i)
        deque_list.pop()
print(time.clock() - start_time, "seconds")
```


```python
import time
start_time = time.clock()
just_list = []
for i in range(10000):
    for i in range(10000):
        just_list.append(i)
        just_list.pop()
print(time.clock() - start_time, "seconds")
```

### OrderedDict
---
- Dict와 달리, 데이터를 입력한 순서대로 dict를 반환
- 잘 안쓸듯

### namedtuple
---
- Tuple 형태로 Data 구조체를 저장하는 방법
- 저장되는 data의 variable을 사전에 지정해서 저장함

## Pythonic code
---
- 파이썬 스타일의 코딩 기법
- 파이썬 특유의 문법을 활용하여 효율적으로 코드를 표현함
- 고급 코드를 작성할 수록 더 많이 필요해짐

### split & join
---


```python
items = 'one two three four'.split()
print(items)
items_join = ' '.join(items)
print(items_join)
```

### list comprehension
---



```python
result = []
for i in range(10):
    result.append(i)

result
```


```python
result = [i for i in range(10)]
result
```


```python
result = [i for i in range(10) if i % 2 ==0]
result
```


```python
word_1 = 'hello'
word_2 = 'world'
result = [i+j for i  in word_1 for j in word_2]
result
```


```python
case_1 = ['a','b','c']
case_2 = ['d','b','f']

result = [i+j for i in case_1 for j in case_2 if not(i==j)]
print(result)
```


```python
words = 'The quick brown fox jumps over the lazy dog'.split()

print(words)

stuff = [[w.upper(),w.lower(),len(w)]for w in words]

for i in stuff:
    print(i)
```

### enumerate & zip


```python
for i, v in enumerate(['tic', 'tac', 'toe']):
    print(i,v)
```

    0 tic
    1 tac
    2 toe
    


```python
alist = ['a1','a2','a3']
blist = ['b1','b2','b3']
for a,b in zip(alist,blist): # 병렬적으로 값을 추출
    print(a,b)
```

    a1 b1
    a2 b2
    a3 b3
    


```python
[sum (x) for x in zip((1,2,3), (10,20,30), (100,200,300))]
```




    [111, 222, 333]



### lambda & map & reduce
---

- lambda : 함수이름 없이, 함수처럼 쓸 수 있는 익명 함수


```python
def f(x, y):
    return x + y

print(f(1, 4))

f = (lambda x, y: x + y)
print(f(10, 50))
(lambda x, y: x + y)(10, 50)
```

    5
    60
    




    60



#### lambda 문제점
    - 어려운 문법
    - 테스트의 어려움
    - 문서화 docstring 지원 미비
    - 코드 해석이 어려움
    - 그래도 많이 쓴다 ..


```python
ex = [1,2,3,4,5]
f = lambda x, y: x + y
print(list(map(f, ex, ex)))
```

    [2, 4, 6, 8, 10]
    


```python
ex = [1,2,3,4,5]
print(list(map(lambda x: x+x, ex))) # 이렇게 리스트를 붙여줘야됨
print((map(lambda x: x+x, ex)))

f = lambda x: x **2
print(map(f,ex))
for i in map(f,ex):
    print(i)
```

    [2, 4, 6, 8, 10]
    <map object at 0x000001E63E2C9940>
    <map object at 0x000001E63E2C9898>
    1
    4
    9
    16
    25
    

#### reduce
---
- map function과 달리 list에 똑같은 함수를 적용해서 통합


```python
from functools import reduce
reduce(lambda x,y : x+ y,[1,2,3,4,5])
```




    15



#### literable object
---
- 내부적 구현으로 \_\_iter__와 \_\_next__가 사용됨
- iter()와 next() 함수로 iterable 객체를 iterator object로 사용


```python
cities = ['Seoul','Busan','Jeju']
cities
```




    ['Seoul', 'Busan', 'Jeju']




```python
memory_address_cities =  iter(cities)
memory_address_cities
```




    <list_iterator at 0x1e63e2aaa90>




```python
next(memory_address_cities)
```




    'Seoul'




```python
next(memory_address_cities)
```




    'Busan'




```python
next(memory_address_cities)
```




    'Jeju'



### generator
--- 
- iterable object를 특수한 형태로 사용해주는 함수
- element가 사용되는 시점에 값을 메모리에 반환   
    : yield를 사용해 한번에 하나의 element만 반환함


```python
def general_list(value):
    result = []
    for i in range(value):
        result.append(i)
    return result
```


```python
general_list(10)
```




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]




```python
def geneartor_list(value):
    result = []
    for i in range(value):
        yield i
```


```python
for a in geneartor_list(10):
    print(a)
    
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    

### generator comprehension
---
- list comprehension과 유사한 형태로 generator 형태의 list 생성
- generator expression 이라는 이름으로도 부름
- [] 대신 ()를 사용하여 표현


```python
gen_ex = (n*n for n in range(500))
print(type(gen_ex))
```

    <class 'generator'>
    

### generator 를 언제쓰냐
---
- list 타입의 데이터를 반환해주는 함수는 generator 로 만듦  
  : 읽기 쉬운 장점, 중간 과정에서 loop 가 중단될 수 있을 때
- 큰 데이터를 처리할 때는 generator expression을 고려하라  
   : 데이터가 커도 처리의 어려움이 없음
- 파일 데이터를 처리할 때도 generator를 쓰자
  

## function passing arguments
---
- 함수에 입력되는 arguments의 다양한 형태
    - Keyword arguments
    - Default arguments
    - Variable-lenght arguments

**Keyword arguments**


```python
def print_somthing(my_name, your_name):
    print("Hello {0}, My name is {1}".format(your_name, my_name))
print_somthing("Sungchul", "TEAMLAB")
print_somthing(your_name="TEAMLAB", my_name="Sungchul")
```

    Hello TEAMLAB, My name is Sungchul
    Hello TEAMLAB, My name is Sungchul
    

**Default arguments**
- parameter의 기본 값을 사용, 입력하지 않을 경우 기본 값 출력



```python
def print_somthing_2(my_name, your_name = 'TEAMLAB'):
    print("Hello {0}, My name is {1}".format(your_name, my_name))
print_somthing_2("Sungchul", "TEAMLAB")
print_somthing_2("Sungchul")
```

    Hello TEAMLAB, My name is Sungchul
    Hello TEAMLAB, My name is Sungchul
    

## variable-length asterisk

### 가변인자 using asterisk
---
- 개수가 정해지지 않은 변수를 함수의 parameter로 사용하는 법
- Keyword arguments와 함께, argument 추가가 가능
- Asterisk(\*) 기호를 사용하여 함수의 parameter 를 표시함
- 입력된 값은 tuple type로 사용할 수 있음
- 가변인자는 오직 한 개만 맨 마지막 parameter 위치에 사용가능

### 가변인자 (variable-length)
- 가변인자는 일반적으로 *args를 변수명으로 사용
- 기존 parameter 이후에 나오는 값을 tuple로 저장함



```python
def asterisk_test(a,b,*args):
    print(list(args))
    print(type(args))

asterisk_test(1,2,3,4,5)
```

    [3, 4, 5]
    <class 'tuple'>
    

### 키워드 가변인자(Keyword variable-length)
---
- Parameter 이름을 따로 지정하지 않고 입력하는 방법
- asterisk(*) 두개를 사용하여 함수의 parameter를 표시함
- 입력된 값은 dict type로 사용할 수 있음
- 가변인자는 오직 한개만 기존 가변인자 다음에 사용


```python
def kwargs_test_3(one, two,*args,**kwargs):
    print(one+two+sum(args))
    print(args)
    print(kwargs)


kwargs_test_3(3, 4, 5, 6, 7, 8, 9, first=3, second=4, third=5)
```

    42
    (5, 6, 7, 8, 9)
    {'first': 3, 'second': 4, 'third': 5}
    

### asterisk - unpacking a container
---
- tuple, dict 등 자료형에 들어가 있는 값을 unpacking
- 함수의 입력값, zip 등에 유용하게 사용가능


```python
def func(a, *args):
    print(a,args)
    print(a,*args)
    print(type(args))
func(1, *(2,3,4,5,6))
```

    1 (2, 3, 4, 5, 6)
    1 2 3 4 5 6
    <class 'tuple'>
    

# 피어세션 정리
---
1. 각자 전날 작성한 학습 정리 공유 하며 토론
2. 어제 진행한 과제에 대해 서로 모르는 것에 대한 공유
3. 오늘 할 공부 내용 공유
4. 조교님과 아이스 브레이킹 타임 
