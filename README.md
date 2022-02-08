## General Coding Standards

### Code Readability

- The code easy to follow and understand.
- All surprises and hard to understand code blocks are preceded by comments which explain the code block. 
- The program is written in terms of the problem domain as much as possible rather than in terms of computer-science or programming-language structures.
- There is no unused commented out code. If commenting out code temporally or for reference, place a TODO item before the block.
- Almost all code is restricted into very simple control flow constructs.
- Deeply nested control statements are avoided.
- Comments explain the code’s intent or summarize what the code does, rather than repeating the code or explaining what it does.

### Code Grouping

- Related methods and statements are grouped together.
- Relatively independent groups of statements are moved into their own routines. 
- Blank lines are used to separate code blocks and make code more readable.

### Routines (methods and functions)

- The routines have a name that is easy to understand and contains no ambiguous hard to understand abbreviations.
- Have strong, functional cohesion (i.e. they do one and one thing only.)
- Have loose coupling (i.e. the routine’s connections to other routines are small, intimate, visible, and flexible – which avoids dependencies and fosters re-usability).
- Have fewer than 7 parameters being passed in (in order to avoid bugs/complexity).
- Each parameter that is passed in is being used.
- No function should be longer than what can be printed on a single sheet of paper. Typically, this means no more than about 100 lines of code per function. 

### Variables

- Are declared close to where they're used.
- The code initializes variables as they’re declared.
- Variables have the smallest scope possible.
- Each variable has only one and only one purpose, and isn’t being re-used for different purposes throughout the code base.
- The variable name fully and accurately describes what the variable represents.
- Constant declarations are used for variables whose values do not change.

### Data/Numbers:

- The code avoids magic numbers. Constants are used in favor of literals (i.e. instead of using 3.1415 within the code, declare a constant called ‘PI’ and use this instead). 
- The code avoids/anticipates divide-by-zero errors.
- The type conversions are obvious, and are avoided.
- Mixed type comparisons are avoided. 
- All over-flow problems are analyzed and anticipated.
- Floating point numbers are not used for monetary computations.
- Floating point numbers are not compared for equality.
- Floating point numbers are avoided as often as possible when exact decimal precision is necessary!
- Hard coding of strings is avoided. Instead, constants or external resources are used. 
- The use of null values or null assignment is avoided as often as possible.

### Conditionals (if-else and case statements):

- The normal case follows the if rather than the else statement.
- The most common cases are tested first.
- If statements are structured in such a way as to keep them as straightforward and understandable as possible. 
- Complicated test cases are encapsulated into boolean function calls.
- All conditional cases are covered.
- Nested if-else conditionals are avoided as much as possible. Nested if-else or case statements are moved into their own routines instead.

### Error/Exception Handling:

- Make sure that the code compiles with no warnings.
- All data structures / opened resources and connections are cleaned up when your program terminates.
- All exceptions are caught and handled appropriately. In other words – if an exception/error is encountered, and it is a recoverable condition, the program halts gracefully and the appropriate logs and users are kept up to date.
- Hard-coded exposure of error handling is avoided by using standard, declarative exception handling routines.
- All error messages are easy to understand and clear. 

### Refactoring:

- Follow the principles outlined in 'Refactoring: Improving the Design of Existing Code.' In general, getting things done comes first, but not at the cost of quality, so always aim to improve your code - even after project delivery. 
- A great summary of the refactoring principles and book can be found [here](https://github.com/HugoMatilla/Refactoring-Summary).

## Python Specific Practices

### Pythonic Thinking

- Follow the [Pep 8 Style guide](https://www.python.org/dev/peps/pep-0008/)
- Prefer interpolated F-Strings over C-style format strings and str.format:

```python3
name = "Eric"
age = 74
f"Hello, {name}. You are {age}." # 'Hello, Eric. You are 74.'
```

- Write helper functions instead of complex expressions.
- Prefer enumerate over range:

```python3
simple_list = ['item 1', 'item 2']
for i, item in enumerate(simple_list):
    print('Do something with item and index i from our list')
```

- Use zip to process iterators in parallel:

```python3
ls1 = [1, 2, 3]
ls2 = ['a', 'b', 'c']
combined_ls = list(zip(ls1, ls2))
print(combined_ls) # [(1, 'a'), (2, 'b'), (3, 'c')]
```

- Prefer get over in and KeyError to handle missing dictionary keys:

```python3
ages = {'Jim': 30, 'Pam': 28, 'Kevin': 33}
missing_age = ages.get('Tim')
if not missing_age:
    print('We are missing Tim''s age!')
```

- Prefer defaultdict over setdefault to handle missing items in internal state:

```python3
from collections import defaultdict
list_dict = defaultdict(list)
list_dict['first_item'] = ['1', '2']
list_dict['missing']  # Accessing a missing key returns an empty list: []
```

- Never unpack more than three variables when functions return multiple values.
- Prefer raising exceptions to returning None
- Provide optional behavior with keyword arguments:

```python3

def show_list(include_values=True):
    some_dict = {}
    for key, value in some_dict.items():
        if include_values:
            print(f"{key}x {value}")
        else:
            print(key)
```

- Use None and Docstrings to specify dynamic default arguments:

```python3
import datetime

def log(message, when=None):
    """Log a message with a timestamp.

    Args:
        message: Message to print.
        when: datetime of when the message occurred.
            Defaults to the present time.
    """
    when = datetime.now() if when is None else when
    print(f'({when}, {message})')
```

- Try to use keyword-only arguments as much as possible (to enforce clarity):

```python3
from pathlib import Path

def read_and_split_file(file_name, separator=" ", default_file_text="default text"):    
    print(f"Processing file_name={file_name}, separator={separator}")
    file_text = Path(file_name).read_text() if file_name else default_file_text
    return file_text.split(separator)

# Below, we use keyword args to show which variables we are referring to:
read_and_split_file(file_name="some_file.txt", separator=" ")
```

- Define function decorators with functools.wraps
- Use comprehensions instead of map and filter:

```python3
the_old_list = [1,2]
# Don't do this!!
the_new_list = map(lambda x: x+23, filter(lambda x: x>5, the_old_list))
# This is much cleaner!!
the_new_list = [x + 23 for x in the_old_list if x > 5]
```

- Avoid more than 2 control subexpressions in comprehensions.
- Consider generators instead of returning lists:

```python3
def process_log_file_lines(log_file_name, lines_starting_with=""):
    log_file = open(log_file_name,"r")
    # in the below return statement, we return a generator instead of reading the entire file!
    return ((line, len(line)) for line in log_file 
            if line.startswith(lines_starting_with))
```

- Consider generator expressions for large list comprehensions.
- Consider itertools for working with iterators and generators.

```python3
# The itertools.takewhile function: it produces a generator that consumes another generator and stops
# when a given predicate evaluates to False.
import itertools
gen = itertools.takewhile(lambda n: n < 3, itertools.count(1, .5)) #  [1, 1.5, 2.0, 2.5]
sample = [5, 4, 2, 8, 7, 6, 3, 0, 9, 1]
list(itertools.accumulate(sample))  # [5, 9, 11, 19, 26, 32, 35, 35, 44, 45]
# We can merge 2 generators using the chain function:
list(itertools.chain('ABC', range(2))) # ['A', 'B', 'C', 0, 1]
```

- Use namedtuple types for tiny, ummutable data classes:

```python3
import collections

Point = collections.namedtuple('Point', 'x, y')
p = Point(1, 2)
print(p.x)  # 1
print(p.y)  # 2
```

- Prefer using data classes (rather than regular ones) for more complex objects:

```python3
# Data classes are just regular classes that are geared towards storing state, more than contain
# a lot of logic. Every time you create a class that mostly consists of attributes you made a data class.
# What the dataclasses module does is make it easier to create data classes. It takes care of a lot of
# boiler plate for you. This is especially important when your data class must be hashable; this requires
# a __hash__ method as well as an __eq__ method. If you add a custom __repr__ method for ease of
# debugging, that can become quite verbose:
from dataclasses import dataclass

@dataclass(unsafe_hash=True)
class InventoryItem:
    '''Class for keeping track of an item in inventory.'''
    name: str
    unit_price: float
    quantity_on_hand: int = 0

    def total_cost(self) -> float:
        return self.unit_price * self.quantity_on_hand
```

- Accept functions instead of classes for simple interfaces:

```python3
# As an example, many of Python’s built-in APIs allow you to customize behavior by passing in a function. 
#  For example, the list type’s sort method takes an optional key argument that’s used to determine each index’s value for sorting:
names = ['Socrates', 'Archimedes', 'Plato', 'Aristotle']
names.sort(key=len) # ['Plato', 'Socrates', 'Aristotle', 'Archimedes']
```

- Consider composing functionality with mix-in classes:

```python3
import json
# Say that I want a mix-in that provides generic JSON serialization for any class. 
# I can do this by assuming that a class provides a to_dict method:
class JsonMixin:
    @classmethod
    def from_json(cls, data):
        kwargs = json.loads(data)
        return cls(**kwargs)

    def to_json(self):
        return json.dumps(self.to_dict())
```

- Inherit from collections.abc for custom container types:

```python3
# Here, we use collections.abc.Sequence. It's a richer interface than the basic sequence.
# Extending it generates iter(), contains(), reversed(), index(), and count().
import collections
class MyAbcSequence(collections.abc.Sequence):
    def __init__(self, a):
        self.a = a
    def __len__(self):
        return len(self.a)
    def __getitem__(self, i):
        return self.a[i]
```

- Use subprocess to manage child process

```python3
# Decoupling the child process from the parent frees up the parent process to run child 
# processes in parallel. Here, we do this by starting all the child processes together 
# with Popen:
import time
import subprocess

start = time.time()
sleep_procs = []
for _ in range(10):
    proc = subprocess.Popen(["sleep", "1"], shell=True)
    sleep_procs.append(proc)

for proc in sleep_procs:
    proc.communicate

end = time.time()
delta = end - start

print(f"Finished in {delta:.3} seconds")
# The above should finish in around ~2 seconds.
# If these processes ran in sequence, the total delay would be 10 seconds and more.
```

- Use threads for blocking I/O, avoid for parallelism (the GIL limits their use for non-IO tasks).
- Use Lock to prevent data races in threads.
- Use Queue to coordinate work between threads.
- Consider ThreadPoolExecutor when threads are necessary for concurrency:

```python3
import requests
import concurrent.futures

def get_wiki_page_existence(wiki_page_url, timeout = 10):
    response = requests.get(url = wiki_page_url, timeout = timeout)
    return "exists" if response.status_code == 200 else "does not exist"

wiki_page_urls = ["https://en.wikipedia.org/wiki/" + str(i) for i in range(50)]

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for url in wiki_page_urls:
        futures.append(executor.submit(get_wiki_page_existence, wiki_page_url=url))
    for future in concurrent.futures.as_completed(futures):
        print(future.result())
```

- Achieve highly concurrent I/O with coroutines:

```python3
import aiohttp
import asyncio
async def donwload_aio(urls):
    async def download(url):
        print(f"Start downloading {url}")
        async with aiohttp.ClientSession() as s:
            resp = await s.get(url)
            out = url, await resp.read()
        print(f"Done downloading {url}")
        return out
        
    return await asyncio.gather(*[download(url) for url in urls])
```

- Consider concurrent.futures for true parallelism:

```python3
from concurrent.futures import ProcessPoolExecutor
import time

NUMBERS = [
    (1823712, 1924928), (2293129, 1020491),
    (1281238, 2273782), (3823812, 4237281),
    (3812741, 4729139), (1292391, 2123811),
]

def greatest_common_divisor(pair):
    a, b = pair
    low = min(a, b)
    for i in range(low, 0, -1):
        if a % i == 0 and b % i == 0:
            return i
    assert False, 'Not reachable'

def main():
    start = time.time()
    pool = ProcessPoolExecutor(max_workers=2)  # The one change
    results = list(pool.map(greatest_common_divisor, NUMBERS))
    end = time.time()
    delta = end - start
    print(f'Took {delta:.3f} seconds')

if __name__ == '__main__':
    main()
```

- Consider contextlib and with statements for reusable try/finally behavior:

```python3
# For example, say you want to write a file and ensure that it's always closed
# correctly. You can do this by passing open to the with statement. open
# returns a file handle for the as target of with and will close the handle
# when the with block exits.

with open('/tmp/my_output.txt', 'w') as handle:
    handle.write('This is some data!')

# We can also implement our own context manager behavior by using the @contextmanager decorator!
from contextlib import contextmanager

def acquire_resource():
    pass
def release_resource():
    pass

@contextmanager
def managed_resource(*args, **kwds):
    # Code to acquire resource, e.g.:
    resource = acquire_resource(*args, **kwds)
    try:
        yield resource
    finally:
        release_resource(resource)
```

- Use datetime instead of time for local clocks:

```python3
import pytz
import datetime

NYC_ARRIVAL = '2014-05-01 23:33:24'
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

nyc_dt_naive = datetime.strptime(NYC_ARRIVAL, TIME_FORMAT)
eastern = pytz.timezone('US/Eastern')
nyc_dt = eastern.localize(nyc_dt_naive)
utc_dt = pytz.utc.normalize(nyc_dt.astimezone(pytz.utc))
print(utc_dt)
# 2014-05-02 03:33:24+00:00

# Once I have a UTC datetime, I can convert it to San Francisco local time.

pacific = pytz.timezone('US/Pacific')
sf_dt = pacific.normalize(utc_dt.astimezone(pacific))
print(sf_dt)
# 2014-05-01 20:33:24-07:00
```

- Make pickle reliable with copyreg.
- Use decimal when precision is paramount:

```python3
from decimal import Decimal
rate = Decimal('1.45')
seconds = Decimal('222')  # 3*60 + 42
cost = rate * seconds / Decimal('60')
print(cost) # 5.365
```

- Use built-in algorithms and data structures (collections.dequeue, collections.Count, etc...) rather than attempting to implement your own. 

```python3
# Great example of a built-in LRU Cache decorator:
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    return n if n < 2 else fib(n-1) + fib(n-2)

print ( [fib(n) for n in range(10)] )
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

print ( fib.cache_info() )
# CacheInfo(hits=16, misses=10, maxsize=None, currsize=10)
```

- Profile before optimizing. The dynamic nature of Python causes surprising behaviors in its 
runtime performance. Operations you might assume would be slow are actually very fast (e.g., 
string manipulation, generators). Language features you might assume would be fast are actually 
very slow (e.g., attribute accesses, function calls). 

```python3
def test():
    pass
from cProfile import Profile
profiler = Profile()
profiler.runcall(test)
```

- Prefer deque for producer–consumer queues:

```python3
import collections
q = collections.deque()
for i in range(100000):
    q.append(1)
for i in range(100000):
    q.popleft()
```

- Always try to search sorted sequences with bisect.
- Know how to use heapq for priority queues.
- Consider memoryview and bytearray for zero-copy interactions with bytes.

## Code Smells

- Try to make the code human readable as much as possible! If you are having trouble reading it and understand it,
create helper functions to make the code more readable. As an example, the below code could be refactored to:

```python3
file_path = 'delete_label'
if (file_path != 'delete_label' or file_path is not None):
    # Do some file manipulation
    pass

# Instead of using the above, we can introduce a helper function called exists which makes the code clearer:
def exists(file_path):
    return file_path != 'delete_label' or file_path is not None

# Then, we refactor the above to:
if exists(file_path):
    # Do some file manipulation
    pass
```

- Always catch, log, or report exceptions!! Don't ignore them!

```python3
def do_something_func():
    pass

try:
    do_something_func()
except Exception:
    # Never ignore an exceptionn!! Exceptions always have something to tell us: a component of the code which was 
    # supposed to execute didn't, and the least we can do is log it!! 
    pass
```

- For better guidelines and info, make sure to read the book full book [Effective Python](https://effectivepython.com/) as well as go through the examples shown in the example_code folder.
- [Fluent Python](https://www.oreilly.com/library/view/fluent-python/9781491946237/) is also fantastic!

## Effective Python

This repo is a direct fork of the _Effective Python: Second Edition_ repository which you can find here:

[Effective Python](https://github.com/bslatkin/effectivepython)

You can find some great examples from the book in the example_code folder.