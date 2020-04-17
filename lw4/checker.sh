nvcc lu_gpu.cu

for i in {0..499}
do
    in="./tests/input${i}"
    out="./tests/output${i}"
   ./a.out < ${in} > tmp
    if ! diff -u tmp ${out} ; then
        echo ${out}
        echo "failed"
    fi
done
