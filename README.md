# Multithread matrix

## Ogólne informacje
W ramach biblioteki zaimplementowano algorytm do wielowątkowego mnożenia macierzy w języku Rust, a następnie funkcję go implementującą wystawiono w formie biblioteki po stronie języka Python.

Do wystawienia funkcji po stronie Pythona wykorzystano bibliotekę [PyO3](https://pyo3.rs/v0.21.2/).

Przetestowano na platformach:
- `Ubuntu 22.04 LTS` + `Intel® Core™ i7-10875H CPU @ 2.30GHz × 16` (`x86_64`)
- `macOS Sonoma 14.3` + `Apple M1` (`arm64`)

Wykorzystane wersje języków:
- Python: `3.10.x`
- Rust `1.78.0`

## Instalacja pakietu pythonowego
Przed instalacją pakietu pythonowego należy się upewnić, że na komputerze na którym chcemy zainstalować bibliotekę jest dostępny kompilator języka Rust.
Instukcja instalacji znajduje się pod tym [linkiem](https://www.rust-lang.org/tools/install).

W katalogu projektu z plikiem `pyproject.toml` wykonujemy następującą komendę:
```bash
pip3 install .
```
> Jeżeli chcemy zainstalować pakiet w środowisku wirtualnym, należy najpierw aktywować środowisko. Dla środowiska wirtualnego utworzonego z wykorzystaniem `anaconda` należy najpierw stworzyć środowisko komendą `conda create env -n <nazwa>`, a następnie aktywować je `conda activate <nazwa>`

## Algorytm
Zamiast standardowego algorytmu mnożenia macierzy w ramach którego jeden wątek oblicza wartości wszystkich elementów macierzy wyjściowej, zastosowano algorytm w ramach którego w zależności od ilości użytych wątków oraz wielkości macierzy wyjściowej do każdego wątku przydzielane jest `n` elementów do obliczenia.

Po podaniu do funkcji oczekiwanej liczby wątków przeprowadzana jest analiza ile elementów przydzielić do jednego wątku. Możemy wyróżnić trzy sytuacje:
1. **Liczba wątków jest równa liczbie elementów macierzy wyjściowej** - każdemu wątkowi przydzielany jest jeden element do obliczenia. 
2. **Liczba wątków jest większa od liczby elementów macierzy wyjściowej** - liczba wątków ograniczana jest do liczby elementów macierzy wyjściowej i każdemu wątkowi jest przydzielany 1 element.
3. **Liczba wątków jest mniejsza od liczby elementów macierzy wyjściowej** 
   4. **Liczba elementów podzielna przez liczbę wątków** - każdemu wątkowi przydzielane jest x = n / threads_count elementów, gdzie n to całkowita liczba elementów w macierzy wyjściowej, a threads_count to oczekiwana liczba użytych wątków. 
   5. **Liczba elementów niepodzielna przez liczbę wątków** - wynik działania  x = n / threads jest zaokrąglany w górę, a następnie liczba wątków jest korygowana wg następującego wzoru: new_threads_count = n / x + 1.
   
Poniżej na rysunku przedstawiono mnożenie dwóch macierzy kwadratowych o wymiarach 3x3 przy użyciu 5 wątków. Kolor tła elementów macierzy `C` oznacza który wątek jest odpowiedzialny za obliczenie wartości tego elementu.

![visualization](doc/multiplication_vis.png)
## Interfejs i przykłady użycia
Biblioteka dostarcza funkcje zarówno po stronie Rust'a, jak i Pythona.

Po stronie pythona funkcja biblioteczna ma następujący interfejs.
```python
def multiply_f64(a: np.ndarray, b: np.ndarray, num_threads=None) -> np.ndarray:
    """
    Multiplies two matrices of type float64 using multiple threads for parallel computation.

    :param a: First matrix, a 2D NumPy array of type float64.
    :param b: Second matrix, a 2D NumPy array of type float64.
    :param num_threads: Optional; The number of threads to use for the computation.
                        If None, the function will use that number of threads provided by the hardware CPU.
    :return: The result of the matrix multiplication as a 2D NumPy array of type float64.
    :raises ValueError: If the number of columns in the first matrix does not match the number of rows in the second matrix.
    :raises TypeError: If the input arrays are not of type float64.
    """
    pass
```
> UWAGA!
> 
> 1. Funkcja przyjmuje tylko macierze zawierajace liczby zmiennoprzecinkowe.
> 2. Mnożenie wektorów także wymaga zdefiniowania macierzy z użyciem biblioteki numpy. Mnożenie elementów `1D array` nie jest wspierane.

### Przykładowe użycie

**Mnożenie macierzy**
```python
import numpy as np
import multithread_matrix as mm
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[1, 2], [3, 4]], dtype=np.float64)
z = mm.multiply_f64(x, y, num_threads=2)
```

**Mnożenie wektorów**
```python
import numpy as np
import multithread_matrix as mm
x = np.array([1, 2, 3, 4], dtype=np.float64)
y = np.array([5, 6, 7, 8], dtype=np.float64).T
z = mm.multiply_f64(x, y, num_threads=2)
```

Wywołanie przykładów
```bash
python3 python_examples/matrix_mult_simple_examples.py
```

## Wydajność
Wywołanie przykładów po stronie Rust'a
```bash
cargo run --release --package multithread_matrix --bin multithread_matrix_bin
```

Przykadowy rezultat programu:
```text
Available CPU's: 16

"Tiny" matrices 2x2
Single threaded: 633ns
Multi threaded: 195.685µs

"Small" matrices 25x25
Single threaded: 21.479µs
Multi threaded: 431.647µs

"Medium" matrices 100x100
Single threaded: 1.20699ms
Multi threaded: 1.424453ms

"Large" matrices 1000x1000
Single threaded: 1.802089957s
Multi threaded: 510.154303ms

"Enormous" matrices 2000x2000
Single threaded: 44.159244698s
Multi threaded: 6.022341037s
```
> Rezultat wywołania programu `rust_examples/main.rs` może się różnić za każdym razem. Jest on zależny od liczby użytych wątków, maszyny, na której jest uruchamiany, oraz aktualnego obciążenia i częstotliwości pracy CPU.

### Porównanie wydajności w trybach jedno i wielowątkowym
Na rysunku poniżej przedstawiono uśrednione na 10 próbach wartości czasu wykonania mnożenia dla trybów jedno i wielowątkowego w zależności od wielkości macierzy kwadratowych.
![comparison](doc/multiplication_performance_log.png)

Jak można zauważyć na samym początku mnożenie jednowątkowe deklasuje odpowiednik wielowątkowy. Jest to związane ze stałym czasem, niezależnym od ilości używanych wątków, wymaganym do przygotowania mnożenia w trybie wielowątkowym.

Przyrost czasu wykonania w trybie wielowątkowym jest jednak wolniejszy od tego występującego w trybie jednowątkowym i już od około wielkości `120x120` wydajność trybu wielowątkowego zrównuje się z trybem jednowątkowym.

Dalsze zwiększanie wielkości macierzy skutkuje jedynie powiększaniem się różnicy pomiędzy tymi trybami, na korzyść tego wielowątkowego.

Testy wydajnościowe po stronie Pythona:
```bash
python3 python_examples/matrix_mult_performance.py
```
## Autor 
Piotr Patek, ZPR, SEM 24L