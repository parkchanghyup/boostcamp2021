
# 강의 복습
---


### 객체
---
- 실생활에서 일종의 물건, 속성(Attribute)과 행동(Action)을 가짐
- 객체 지향 프로그램은 이러한 객체 개념을 프로그램으로 표현 속성은 변수, 행동은 함수로 표현됨
- 파이썬 역시 **객체 지향 프로그램 언어**
       

### python naming rule
---

- 변수와 Class명 함수명은 짓는 방식이 존재
- snake_case : 띄어쓰기 부분에 "_" 를 추가 ( 파이썬 함수/변수명에 사용)
- CamelCase : 띄어쓰기 부분에 대문자 낙타의 등 모양, 파이썬 Class명에 사용

### Attribute 추가하기
---
- Attribute 추가는 \__init__, self와 함께!
- \__init__은 객체 초기화 예약 함수


```python
class SoccerPlayer(object):
    def __init__(self, name, position, back_number):
        self.name = name
        self.position = position
        self.back_number = back_number
```

### 파이썬에서 __의 의미
---
- __는 특수한 예약 함수나 변수 그리고 함수명 변경(맨글링)으로 사용
- ex) \_\_main__, \_\_add__, \_\_str__

### method 구현하기
---
- method(Action) 추가는 기존 함수와 같으나, 반드시 self를 추가해야만 class 함수로 인정됨

![method](./image/method.PNG)

### object(instance) 사용하기
---
![object](./image/method.PNG)

### 예제 코드
---



```python
class SoccerPlayer(object):
    def __init__(self, name, position, back_number):
        self.name = name
        self.position = position
        self.back_number = back_number
    
    def change_back_number(self,new_number):
        print('선수의 등번호를 변경합니다 : from %d to %d'%(self.back_number,new_number))
        self.back_number = new_number
    
    def __str__(self):
        return 'hello, my name is %s. my back number is %d'%\
    (self.name, self.back_number)
son = SoccerPlayer('son','MF',7)
print(son)
```

    hello, my name is son. my back number is 7
    

## 객체 지향 실습 - 노트북
---
- Note를 정리하는 프로그램
- 사용자는 Note에 뭐가를 적을 수 있다.
- Note에는 Content가 있고, 내용을 제거할 수 있다.
- 두 개의 노트북을 합쳐 하나로 만들 수 있다.
- Note는 Notebook에 삽입된다.
- Notebook은 Note가 삽일 될 때 페이지를 생성하며,최고 300페이지 까지 저장가능하다
- 300 페이지가 넘으면 더 이상 노트를 삽입하지 못한다.


```python
class Note():
    def __init__(self,content = None):
        self.content = content
    
    def write_content(self,content):
        self.content = content

    def remove_all(self):
        self.content =''
    
    def __add__(self, other):
        return self.content +other.content
    
    def __str__(self):
        return '노트에 적힌 내용 입니다 : ' + self.content
    
    

```


```python
class NoteBook(object):
    def __init__(self, title):
        self.title = title
        self.page_number = 1
        self.notes = {}
    
    def add_note(self, note, page = 0):
        if self.page_number < 300:
            if page ==0:
                self.notes[self.page_number] = note
                self.page_number +=1
            else :
                self.notes = {page : note}
                self.page_number +=1
        else :
            print('Page가 모두 채워 졌습니다.')
    
    def remove_note(self, page_number):
        if page_number in self.notes.keys():
            return self.notes.pop(page_number)
        else:
            print('해당 페이지는 존재 하지 않습니다.')
            
    def get_number_of_pages(self):
        return len(self.notes.keys())
```


```python
my_notebook = NoteBook('강의노트')
my_notebook.title
```




    '강의노트'




```python
new_note = Note('파이썬 노노')
print(new_note)
```

    노트에 적힌 내용 입니다 : 파이썬 노노
    


```python
new_note2 = Note('파이썬 예쓰')
print(new_note2)
```

    노트에 적힌 내용 입니다 : 파이썬 예쓰
    


```python
my_notebook.add_note(new_note)
```


```python
print(my_notebook.notes[6])
```

    노트에 적힌 내용 입니다 : 파이썬 노노
    


```python
my_notebook.get_number_of_pages()
```




    3



## 객체 지향 언어의 특징
---
- 상속
- 다형성
- 가시성

### 상속
---
- 부모클래스로 부터 속성과 Method를 물려받은 자식 클래스를 생성 하는것  
  



```python
class Person:
    def __init__(self,name,age,gender):
        self.name = name
        self.age = age
        self.gender = gender
    
    def about_me(self):
        print('저의 이름은',self.name,'이구오, 제나이는',
             str(self.age),' 살 입니다.')
    
    def __str__(self):
        return '저의 이름은 {0} 입니다. 나이는 {1} 입니다.'\
    .format(self.name,self.age)
```


```python
class Korean(Person):
    pass

first_korean = Korean('park',35)
print(first_korean)
```

    저의 이름은 park 입니다. 나이는 35 입니다.
    


```python
class Employee(Person) : # 부모 클래스 Person으로 부터 상속
    def __init__(self ,name, age, gender, salary, hire_date):
        super().__init__(name, age, gender) # 부모 객체 사용
        self. salary = salary
        self.hire_date = hire_date # 속성값 추가
        
    def do_word(self):
        print('열심히 일하자')
        
    def about_me(self):
        super().about_me()
        print('제 급여는 ' , self.salary, '원 이구요, 제 입사일은',
             self.hire_date,'입니다.')
```


```python
myPerson = Person('Johh',34,'Male')
myPerson.about_me()
```

    저의 이름은 Johh 이구오, 제나이는 34  살 입니다.
    


```python
myEmployee = Employee('park',50,'Male','300000','2020/20/20')
myEmployee.about_me()
```

    저의 이름은 park 이구오, 제나이는 50  살 입니다.
     제 급여는  300000 원 이구요, 제 입사일은 2020/20/20 입니다.
    

### 다형성
---
- 같은 이름 메소드의 내부 로직을 다르게 작성
- Dynamic Typin 특성으로 인해 파이썬에서는 같은 부모클래스의 상속에서 주로 발생함
- 중요한 OOP의 개념 그러나 너무 깊이 알 필요는 없다.


```python
class Animal:
    def __init__(self, name): # Constructor of the class
        self.name = name
        
    def talk(self): # Abstract method, defined by convention only
        raise NotImplementedError("Subclass must implement abstract method")

class Cat(Animal):
    def talk(self):
        return 'Meow!'
    
class Dog(Animal):
    def talk(self):
        return 'Woof! Woof!'

    
animals = [Cat('Missy'),
Cat('Mr. Mistoffelees'),
Dog('Lassie')]

for animal in animals:
    print(animal.name + ': ' + animal.talk())
```

    Missy: Meow!
    Mr. Mistoffelees: Meow!
    Lassie: Woof! Woof!
    

### 가시성
---
- 객체의 정보를 볼 수 있는 레벨을 조절하는 것
- 누구나 객체 안에 모든 변수를 볼 필요가 없음

    1. 객체를 사용하는 사용자가 임의로 정보 수정
    2. 필요 없는 정보에는 접근 할 필요가 없음
    3. 만약 제품으로 판매한다면? 소스의 보호

### 가시성 예제 1
---
- Product 객체를 Inventory 객체에 추가
- Inventory에는 오직 Product 객체만 들어감
- Inventory에 Product가 몇 개인지 확인이 필요
- Inventory에 Product items는 직접 접근이 불가


```python
class Product:
    pass

class Inventory():
    def __init__(self):
        self.__items = []
    
    def add_new_item(self,product):
        if type(product) == Product:
            self.__items.append(product) # private 변수로 지정
            print('new item added')
        else:
            raise ValueError('Invalid Item')
    
    def get_number_of_items(self):
        return len(self.__items)
```

### 가시성 예제 2
---
- Product 객체를 Inventory 객체에 추가
- Inventory에는 오직 Product 객체만 들어감
- Inventory에 Product가 몇 개인지 확인이 필요
- Inventory에 Product items 접근 허용


```python
class Inventory(object):
    def __init__(self):
        self.__items = []
        
    @property # property decorator 숨겨진 변수를 반환하게 해줌
    def items(self):
        return self.__items
    

```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-79-6fb8dbbb6c23> in <module>
         10 items = my_inventory.items
         11 items.append(Product())
    ---> 12 print(my_inventory.get_number_of_items())
    

    AttributeError: 'Inventory' object has no attribute 'get_number_of_items'


## 데코레이터

### First-class objects
---
- 일등함수 or 일급 객체
- 변수나 데이터 구조에 할당이 가능한 객체
- 파라미터로 전달이 가능 + 리턴 값으로 사용


**파이썬의 모든 함수는 일급함수이다**


```python
def square(x):
    return x*x

def cube(x):
    return x*x*x

def formula(method, argument_list):
    return [method(value) for value in argument_list]
```

### Inner function
--- 
-함수 내에 또 다른 함수가 존재



```python
def print_msg(msg):
    def printer():
        print(msg)
    return printer

another = print_msg('hello,pthon')
another()
```

    hello,pthon
    

### closure example


```python
def tag_func(tag, text):
    text = text
    tag = tag
    
    def inner_func():
        return '<{0}>{1}<{0}>'.format(tag, text)
    
    return inner_func

h1_func = tag_func('title', "This is Python Class")
p_func = tag_func('p', "Data Academy") 
h1_func
```




    <function __main__.tag_func.<locals>.inner_func()>



### decorator function
---
- 복잡한 클로져 함수를 간단하게!



```python
def star(func):
    def inner(*args, **kwargs):
        print('*' * 30)
        func(*args, **kwargs)
        print('*' * 30)
    return inner
@star
def printer(msg):
    print(msg)
printer('hello')
```

    ******************************
    hello
    ******************************
    


```python
def generate_power(exponent):
    def wrapper(f):
        def inner(*args):
            result = f(*args)
            return exponent**result
        return inner
    return wrapper

@generate_power(2)
def raise_two(n):
    return n**2

print(raise_two(7))
```

    562949953421312
    

## Module
---
- 프로그램에서는 작은 프로그램 조각들, 
모듈들을 모아서 하나의 큰프로그램을 개발함
- 프로그램을 모듈화 시키면 다른 프로그램을 사용하기 쉬움
    - ex. 카카오톡 게임을 위한 카카오톡 접속 모듈

### Module 만들기
---
- 파이썬의 Module ==py 파일을 의미
- 같은 폴더에 Module에 해당하는 .py파일과 사용하는 .py을 저장한 후
- import 문을 사용해서 module를 호출


```python
fah_converter.py
```


```python
def covert_c_to_f(celcius_value):
    return celcius_value * 9 / 5 + 32
```


```python
module_ex.py
```


```python
import fah_converter

print('enter a celsius value: ')
celsius = float(input())
fahrenheit = fah_converter.covert_c_to_f(celsius)
print('that's', fahrenheit ','degrees fahrenheit')
```

### namespace
---
- 모듈을 호출할 때 범위 정하는 방법
- 모듈 안에는 함수와 클래스 등이 존재 가능
- 필요한 내용만 골라서 호출할 수 있음
- from과 import 키워드를 호출함

### Alias 설정하기 - 모듈명을 별칭으로 써서
---


```python
import fah_converter as fah
print(fah.covert_c_to_f(41.6))
```

### 모듈에서 특정 함수 또는 클래스만 호출하기
---


```python
from fah_converter import covert_c_to_f
print(covert_c_to_f(41.6))
```

### 모듈에서 모든 함수 또는 클래스를 호출하기
---



```python
from fah_converter import *
print(covert_c_to_f(41.6)
```

## 실습: 1부터 100까지 특정 난수 뽑기
---


```python
import random 
random.randint(1,100)
```




    10



## 패키지
---
- 하나의 대형 프로젝트를 만드는 코드의 묶음
- 다양한 모듈들의 합,폴더로 연결됨
- \__init__,\__main__등 키워드 파일명이 사용됨
- 다양한 오픈 소스들이 모두 패키지로 관리됨

## package 만들기
---
![package](./image/package.PNG)

# 피어세션
---
- 어제 했던 공부 토론
- 각자 학습 정리에 대한 피드백(?),도움
-오늘 공부 할 내용 + 오늘 과제에 대한 토론


```python

```


```python

```


```python

```
