Bootstrap: docker
From: lcazenille/qdpy-bipedal_walker


%labels
    Author leo.cazenille@gmail.com
    Version 0.1.0

%environment
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8

%post
    pip3 uninstall -y qdpy
    pip3 install --upgrade --no-cache-dir git+https://gitlab.com/leo.cazenille/qdpy.git@develop

%runscript
    echo "Nothing there yet..."

%apprun main
    exec ./scripts/main.py $@

%apprun several_main
    nb_runs=$1
    shift;
    for i in $(seq 1 $nb_runs); do
        sleep 1
        ./scripts/main.py $@
    done

%apprun parallel_main
    nb_runs=$1
    shift;
    for i in $(seq 1 $nb_runs); do
        sleep 1
        logname=$(date +%Y%m%d%H%M%S).log
        ./scripts/main.py $@ 2&>1 > results/$logname &
    done
    wait


