These files are for the verification and synthesis of fault-tolerant programs.

Here is a listing of the files:
- ```fallible_parser.py```: A runnable parser which will parse one of the example benchmarks and output the invariants and necessary constraints in both Sketch and SyGuS. It will then begin to run the CVC5 solver on the problem. The CVC5 solver was only successful on the first benchmark, "atomic write" (```python3 fallible_parser.py aw```).
- ```inv_swap_transfer.sk```: A Sketch file for verifying benchmark 2 (atomic arithmetic), which has been modified to provide a partial form for the invariant to synthesize.
- ```inv_synth_swap.sk```: A Sketch file for verifying benchmark 3 (swap registers), which has been modified to provide a partial form for the invariant to synthesize.
- ```fallible_madd.sk```: A Sketch file for synthesizing an atomic multiply-add function, using no temporary intermediate state.
- ```falible_block.sk```: A Sketch file for writing an arithmetic sequence to a block of memory. Ideally, the solver should find a while loop which satisfies the examples, but the solver fails because the problem is too large.