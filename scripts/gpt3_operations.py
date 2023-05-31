import openai
import json

def qa_generator(paragraph:str, temperature:float, model_engine):
    question = f'''
        text: {paragraph}
    '''
    expertise = '''
    Analyse the given text and create questions and answers based on this text
    Present the questions and answers in a JSON format, like this:
    [{ question: 'text_question', answer: 'text_answer'}, { question: 'text_question', answer: 'text_answer'} . . .]
                    '''

    mess = [
            {"role": "system", "content": expertise},
            {"role": "user", "content": question},
        ]
    
    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=mess,
        temperature=temperature,
    )
    response_str = response["choices"][0]["message"]["content"]
    response_str = response_str.replace("'", '"')
    response_dict = json.loads(response_str)
    result = {'prompt':expertise,
              'context':paragraph,
              'qa':response_dict
              }
    return result