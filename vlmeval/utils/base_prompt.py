
def create_one_example(format_, question, context, options, answer, knowledge, image_path):
    
    input_format, output_format = format_.split("-")
    
    aff_base = "You are a fellow debater from the AFFIRMATIVE side, You are more Emotional to think about problems."
    neg_base = "You are a fellow debater from the NEGATIVE side, You are more Rational in thinking about problems."

    kg_emo = f"Emotional Graph: {knowledge[0]}\n" if knowledge[0]!='none' else ""
    kg_rat = f"Rational Graph: {knowledge[1]}\n" if knowledge[1]!='none' else ""
    
    hint = f"Hint: {context}\n" if context != "none" else ""
    question_ = f"Question: {question}\n"
    option_ = f"Options:\n{options}" if options != "none" else ""
    answer_ = "Please select the correct answer from the options above, without any explaination." if options != "none" else "Answer directly."
    
    if input_format=="IQ":
        input = f"""{hint}{question_}{option_}{answer_}"""

    elif input_format == "QIM":
        input = f"""{hint}{question_}{option_}For the provided image and its associated question, generate a graph to answer the question.
"""
    
    elif input_format == "GKG":
        input = f"""For the provided image and its associated question. generate a scene graph in JSON format that includes the following:
1. Objects, attributes, relationships that are more relevant to answering the question.
2. Objects are NO MORE THAN 3. 
{hint}{question_}"""

    #### ODebate_stage
    elif input_format == "ODIM":
        input = f"""{hint}{question_}You are a fellow debater from the AFFIRMATIVE side, You are more Emotional to think about problems.
For the provided image and its associated question, Do not give the answer, But your solution and ideas to solve this question.
""" 
        
    elif input_format == "ONIM":
        input = f"""{hint}{question_}You are a fellow debater from the NEGATIVE side, You are more Rational in thinking about problems.  
For the provided image and its associated question, Do not give the answer, But your solution and ideas to solve this problem. 
"""   
    elif input_format == "ODQIM":
        input = f"""{hint}{question_}Debate Solution:{knowledge[1]}\nYou are a fellow debater from the AFFIRMATIVE side, You are more Emotional to think about problems. 
Based on the debate Solution of the question, Do not give the answer, But your Better solution and ideas to solve this problem.
""" 
        
    elif input_format == "ONQIM":
        input = f"""{hint}{question_}Debate Solution:{knowledge[0]}\nYou are a fellow debater from the NEGATIVE side, You are more Rational in thinking about problems. 
Based on the debate Solution of the question, Do not give the answer, But your Better solution and ideas to solve this problem. 
""" 
        
    elif input_format == "OAGM":
        input = f"""You're good at summarizing and answering questions. \nEmotional Solution: {knowledge[0]}\nRational Solution: {knowledge[1]}\n{hint}{question_}{option_}{answer_}
""" 
    
    
        #### Debate_KG_stage
    elif input_format == "KDIM":
        input = f"""{aff_base}
For the provided image and its associated question, Please give your solution and ideas to solve this problem, but do not give a final answer. Generate an updated graph from a different view based on the Debate Graph in JSON format that includes the following:
1. Objects, attributes, relationships that are more relevant to answering the question.
2. Objects are NO MORE THAN 3.  
{hint}{question_}""" 
        
    elif input_format == "KNIM":
        input = f"""{neg_base}  
For the provided image and its associated question, Please give your solution and ideas to solve this problem, but do not give a final answer. Generate an updated graph from a different view based on the Debate Graph in JSON format that includes the following:
1. Objects, attributes, relationships that are more relevant to answering the question.
2. Objects are NO MORE THAN 3. 
{hint}{question_}"""  
        
    elif input_format == "KDQIM":
        input = f"""{aff_base} 
For the provided image and its associated question, Please give your solution and ideas to solve this problem, but do not give a final answer. Generate an updated graph from a different view based on the Debate Graph in JSON format that includes the following:
1. Objects, attributes, relationships that are more relevant to answering the question.
2. Delete the irrelevant objects, attributes and relationships.
{hint}{question_}{kg_rat}
""" 
        
    elif input_format == "KNQIM":
        input = f"""{neg_base} 
For the provided image and its associated question, Please give your solution and ideas to solve this problem, but do not give a final answer. Generate an updated graph from a different view based on the Debate Graph in JSON format that includes the following:
1. Objects, attributes, relationships that are more relevant to answering the question.
2. Delete the irrelevant objects, attributes and relationships.
{hint}{question_}{kg_emo}
""" 
        
    elif input_format == "KAGM":
        input = f"""You're good at summarizing and answering questions.{hint}{kg_emo}{kg_rat}Use the image and two debate Solution as context and answer the following question:\n{question_}{option_}{answer_}
"""
        
    # Outputs
    if output_format == 'A':
        output = "Answer:"
        
    elif output_format == 'G':
        output = f"Solution: "

    text = input + output
    text = text.replace("  ", " ")
    if text.endswith("BECAUSE:"):
        text = text.replace("BECAUSE:", "").strip()
    return text
