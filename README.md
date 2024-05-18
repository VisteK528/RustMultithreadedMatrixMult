# Python'owa biblioteka do wielowątkowego mnożenia macierzy z backend'em w języku Rust

## Ogólne informacje
Biblioteka dostarcza po stronie Pythona nastepującą funkcję:
```python
def multiply_f64(a: np.ndarray, b: np.ndarray, num_threads=None) -> np.ndarray:
    ...
```
- `a`, `b` - macierze / wektory zdefiniowane z wykorzystaniem biblioteki numpy
- `num_threads` - ilość wątków użytych do mnożenia. Domyślnie wykorzystuje tyle wątków ile fizycznych CPU (jaka równoległość jest realna do osiągnięcia)

Przykadowy rezultat programu:
```text
Available CPU's: 16
Small matrices - Single threaded: 2.129µs
Small matrices - Multi threaded: 334.109µs
Equal?: true
Large matrices - Single threaded: 179.966159ms
Large matrices - Multi threaded: 53.159165ms
Equal?: true
```

### W jaki sposób wykonywane jest mnożenie wielowątkowo?
W zależności od wielkości macierzy `a` i `b` mamy różną ilość elementów w wyknikowej macierzy `c`. Po podaniu ilości wykorzystywanych wątków elementy macierzy `c` do obliczenia są dzielone po równo pomiędzy wszystkie wątki.

## Struktura projektu

## Instalacja pakietu pythonowego
W katalogu projektu wykonujemy następującą komendę:
```bash
pip3 install .
```

Wykorzystywane biblioteki:
- `numpy` - definiowanie macierzy
- `maturin` - budowa projektu

## Biblioteka w języku Rust
Uruchamianie testów jednostkowych w katalogu projektu
```bash
cargo test
```

Uruchomienie przykładów w Rust
```bash
cargo run --release --package zpr_multithread_matrix --bin zpr_multithread_matrix_bin
```