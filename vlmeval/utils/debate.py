from vlmeval.utils import init_prompt_multi

def Debate_VLM(stage, model, struct, dataset_name, debate, kg_init, logger):
    if stage[:8] == "baseline":
        prompt_format = "IQ-A"
        prompt_G = init_prompt_multi(struct, prompt_format)
        response = model.generate(prompt=prompt_G, image_path=struct['image'], dataset=dataset_name)
        logger.info("########--G_A--######\nPrompt: {}\nGT: {} - ANS: {}".format(prompt_G, struct['text']['answer'], response))

    elif stage[:7] == "ODebate":
        for debate_ in range(debate):
            logger.info("########--DEBATE{}--######".format(debate_))

            prompt_format = "ODIM-S" if debate_==0 else "ODQIM-S"
            prompt_A = init_prompt_multi(struct, prompt_format)
            kg_aff = model.generate(prompt=prompt_A, image_path=struct['image'], dataset=dataset_name)

            prompt_format = "ONIM-S" if debate_==0 else "ONQIM-S"
            prompt_N = init_prompt_multi(struct, prompt_format)
            kg_neg = model.generate(prompt=prompt_N, image_path=struct['image'], dataset=dataset_name)

            struct['kg'] = [kg_aff.strip(), kg_neg.strip()]
            prompt_format = "OAGM-A"
            prompt_F = init_prompt_multi(struct, prompt_format)
            response = model.generate(prompt=prompt_F, image_path=struct['image'], dataset=dataset_name, max_length=20)

            logger.info("########--ANSWER-{}--######\n{}".format(debate_, response))

        logger.info("\nGT:{}-ANS: {} - ".format(struct['text']['answer'], response))

    elif stage[:7] == "BDebate":
        for debate_ in range(debate):
            logger.info("########--DEBATE{}--######".format(debate_))
            if debate_ == 0:
                if kg_init:
                    if struct['text']['kg'] != 'none':
                        logger.info("#####---KG_IB---#####\n{}".format(struct['text']['kg']))
                        struct['kg'] = [struct['text']['kg'], struct['text']['kg']]
                    else:
                        prompt_format = "GKG-G"
                        prompt_G = init_prompt_multi(struct, prompt_format)
                        kg_base = model.generate(prompt=prompt_G, image_path=struct['image'], dataset=dataset_name)
                        struct['kg'] = [kg_base, kg_base]
                        logger.info("#####---KG_P---#####\n{}".format(prompt_G))
                        logger.info("#####---KG_B---#####\n{}".format(kg_base))
                else:
                    prompt_format = "GKG-G"
                    prompt_G = init_prompt_multi(struct, prompt_format)
                    kg_base = model.generate(prompt=prompt_G, image_path=struct['image'], dataset=dataset_name)
                    struct['kg'] = [kg_base, kg_base]
                    logger.info("#####---KG_P---#####\n{}".format(prompt_G))
                    logger.info("#####---KG_B---#####\n{}".format(kg_base))

            prompt_format = "KDQIM-G"
            prompt_A = init_prompt_multi(struct, prompt_format)
            kg_aff = model.generate(prompt=prompt_A, image_path=struct['image'], dataset=dataset_name)
            struct['kg'] = [kg_aff,struct['kg'][1]]

            prompt_format = "KNQIM-G"
            prompt_N = init_prompt_multi(struct, prompt_format)
            kg_neg = model.generate(prompt=prompt_N, image_path=struct['image'], dataset=dataset_name)

            struct['kg'] = [kg_aff, kg_neg]
            prompt_format =  "KAGM-A"
            prompt_F = init_prompt_multi(struct, prompt_format)
            response = model.generate(prompt=prompt_F, image_path=struct['image'], dataset=dataset_name)
            
            logger.info("########--ANSWER-{}--######\n{}".format(debate_, response))

        logger.info("\nGT:{}-ANS: {} - ".format(struct['text']['answer'], response))

    else:
        assert stage == "BDebate", f"Please confirm if your 'stage' is set correctly.\nDebate Only: Begin with ODebate\nDebate with Blueprint:Begin with BDebate\nStage setting Now:{stage}"
    return response
        