torchrun --nproc_per_node=2 run.py --data ScienceQA_TEST \
                                   --model instructblip_13b \
                                   --stage BDebate_kg_test \
                                   --debate 2
                                   --kg_init
