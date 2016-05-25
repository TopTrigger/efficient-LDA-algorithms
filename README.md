# efficient-LDA-algorithms

Run single experiment with the code
```
python efficient-lda.py <lda_type> <K> <max_iter> <log name>
```
where 
  * `<lda_type>` can be 'baseline', 'sparse' and 'alias'
  * `<K>` is the number of topics
  * `<max_iter>` designates the maximum iteration number
  * `<log name>` is the path to log file. Time information and loglikelihood will be written into the log file.

Run all the experiments with the code
```
python shell.py
```
where `shell.py` implements all the combinations of different LDA algorithms and different LDA algorithms
