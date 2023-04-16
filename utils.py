import numpy as np
import random
import torch
import os

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


mental_health_groups = [
    'EDAnonymous',
    'addiction',
    'alcoholism',
    'adhd',
    'anxiety',
    'autism',
    'bipolarreddit',
    'bpd',
    'depression',
    'healthanxiety',
    'lonely',
    'ptsd',
    'schizophrenia',
    'socialanxiety',
    'suicidewatch'
]

non_mental_health = [
    'conspiracy',
    'divorce',
    'fitness', 
    'guns', 
    'jokes', 
    'legaladvice', 
    'meditation', 
    'parenting', 
    'personalfinance', 
    'relationships', 
    'teaching',
]

len(mental_health_groups) + len(non_mental_health)

mental_health_description = {
    "EDAnonymous": "An eating disorder is a mental disorder defined by abnormal eating behaviors that negatively affect a person's physical or mental health. Types of eating disorders include binge eating disorder, where the patient eats a large amount in a short period of time; anorexia nervosa, where the person has an intense fear of gaining weight and restricts food or overexercises to manage this fear; bulimia nervosa, where individuals eat a large quantity (binging) then try to rid themselves of the food (purging); pica, where the patient eats non-food items; rumination syndrome, where the patient regurgitates undigested or minimally digested food; avoidant/restrictive food intake disorder (ARFID), where people have a reduced or selective food intake due to some psychological reasons; and a group of other specified feeding or eating disorders. Anxiety disorders, depression and substance abuse are common among people with eating disorders. These disorders do not include obesity.",
    "addiction": "Addiction is generally a neuropsychological symptom defining pervasive and intense urge to engage in maladaptive behaviors providing immediate sensory rewards (e.g. consuming drugs, excessively gambling), despite their harmful consequences. Dependence is generally an addiction that can involve withdrawal issues. Addictive disorder is a category of mental disorders defining important intensities of addictions or dependences, which induce functional disabilities.",
    "adhd": "Attention deficit hyperactivity disorder (ADHD) is a neurodevelopmental disorder characterised by excessive amounts of inattention, hyperactivity, and impulsivity that are pervasive, impairing in multiple contexts, and otherwise age-inappropriate.",
    "alcoholism": "Alcoholism is, broadly, any drinking of alcohol that results in significant mental or physical health problems. Because there is disagreement on the definition of the word alcoholism, it is not a recognized diagnostic entity, and the use of alcoholism terminology is discouraged due to its heavily stigmatized connotations. Predominant diagnostic classifications are alcohol use disorder (DSM-5) or alcohol dependence (ICD-11).",
    "anxiety": "Anxiety is an emotion which is characterized by an unpleasant state of inner turmoil and includes feelings of dread over anticipated events. Anxiety is different than fear in that the former is defined as the anticipation of a future threat whereas the latter is defined as the emotional response to a real threat. It is often accompanied by nervous behavior such as pacing back and forth, somatic complaints, and rumination.",
    "autism": "The autism spectrum, often referred to as just autism, autism spectrum disorder (ASD) or sometimes autism spectrum condition (ASC), identifies a loosely defined cluster of neurodevelopmental disorders characterized by challenges in social interaction, verbal and nonverbal communication, and often repetitive behaviors and restricted interests. Other common features include unusual responses to sensory stimuli and a preference for sameness or unusual adherence to routines.",
    "bipolarreddit": "Bipolar disorder, previously known as manic depression, is a mental disorder characterized by periods of depression and periods of abnormally elevated mood that each last from days to weeks. If the elevated mood is severe or associated with psychosis, it is called mania; if it is less severe, it is called hypomania. During mania, an individual behaves or feels abnormally energetic, happy or irritable, and they often make impulsive decisions with little regard for the consequences. There is usually also a reduced need for sleep during manic phases. During periods of depression, the individual may experience crying and have a negative outlook on life and poor eye contact with others. The risk of suicide is high. Other mental health issues, such as anxiety disorders and substance use disorders, are commonly associated with bipolar disorder.",
    "bpd": "Borderline personality disorder (BPD), also known as emotionally unstable personality disorder (EUPD), is a personality disorder characterized by a long-term pattern of intense and unstable interpersonal relationships, distorted sense of self, and strong emotional reactions. Those affected often engage in self-harm and other dangerous behaviors, often due to their difficulty with returning their emotional level to a healthy or normal baseline. They may also struggle with a feeling of emptiness, fear of abandonment, and detachment from reality.",
    "depression": "Depression is a mental state of low mood and aversion to activity. It affects more than 280 million people of all ages (about 3.5% of the global population). Depression affects a person's thoughts, behavior, feelings, and sense of well-being. Depressed people often experience loss of motivation or interest in, or reduced pleasure or joy from, experiences that would normally bring them pleasure or joy. Depressed mood is a symptom of some mood disorders such as major depressive disorder and dysthymia; it is a normal temporary reaction to life events, such as the loss of a loved one; and it is also a symptom of some physical diseases and a side effect of some drugs and medical treatments. It may feature sadness, difficulty in thinking and concentration and a significant increase or decrease in appetite and time spent sleeping. People experiencing depression may have feelings of dejection or hopelessness and may experience suicidal thoughts.",
    "healthanxiety": "Hypochondriasis or hypochondria is a condition in which a person is excessively and unduly worried about having a serious illness. Hypochondria is an old concept whose meaning has repeatedly changed over its lifespan. It has been claimed that this debilitating condition results from an inaccurate perception of the condition of body or mind despite the absence of an actual medical diagnosis. An individual with hypochondriasis is known as a hypochondriac. Hypochondriacs become unduly alarmed about any physical or psychological symptoms they detect, no matter how minor the symptom may be, and are convinced that they have, or are about to be diagnosed with, a serious illness.",
    "lonely": "Loneliness is an unpleasant emotional response to perceived isolation. Loneliness is also described as social pain – a psychological mechanism which motivates individuals to seek social connections. It is often associated with a perceived lack of connection and intimacy. Loneliness overlaps and yet is distinct from solitude. Solitude is simply the state of being apart from others; not everyone who experiences solitude feels lonely. As a subjective emotion, loneliness can be felt even when a person is surrounded by other people. Hence, there is a distinction between being alone and feeling lonely. Loneliness can be short term (state loneliness) or long term (chronic loneliness). In either case, it can be intense and painful.",
    "ptsd": "Post-traumatic stress disorder (PTSD) is a mental and behavioral disorder that can develop because of exposure to a traumatic event, such as sexual assault, warfare, traffic collisions, child abuse, domestic violence, or other threats on a person's life. Symptoms may include disturbing thoughts, feelings, or dreams related to the events, mental or physical distress to trauma-related cues, attempts to avoid trauma-related cues, alterations in the way a person thinks and feels, and an increase in the fight-or-flight response. These symptoms last for more than a month after the event. Young children are less likely to show distress but instead may express their memories through play. A person with PTSD is at a higher risk of suicide and intentional self-harm.",
    "schizophrenia": "Schizophrenia is a mental disorder characterized by continuous or relapsing episodes of psychosis. Major symptoms include hallucinations (typically hearing voices), delusions, and disorganized thinking. Other symptoms include social withdrawal, decreased emotional expression, and apathy. Symptoms typically develop gradually, begin during young adulthood, and in many cases never become resolved. There is no objective diagnostic test; diagnosis is based on observed behavior, a psychiatric history that includes the person's reported experiences, and reports of others familiar with the person. To be diagnosed with schizophrenia, symptoms and functional impairment need to be present for six months (DSM-5) or one month (ICD-11). Many people with schizophrenia have other mental disorders, especially substance use disorders, depressive disorders, anxiety disorders, and obsessive–compulsive disorder.",
    "socialanxiety": "Social anxiety is the anxiety and fear specifically linked to being in social settings (i.e., interacting with others). Some categories of disorders associated with social anxiety include anxiety disorders, mood disorders, autism spectrum disorders, eating disorders, and substance use disorders. Individuals with higher levels of social anxiety often avert their gazes, show fewer facial expressions, and show difficulty with initiating and maintaining a conversation. Social anxiety commonly manifests itself in the teenage years and can be persistent throughout life, however, people who experience problems in their daily functioning for an extended period of time can develop social anxiety disorder. Trait social anxiety, the stable tendency to experience this anxiety, can be distinguished from state anxiety, the momentary response to a particular social stimulus. Half of the individuals with any social fears meet the criteria for social anxiety disorder. Age, culture, and gender impact the severity of this disorder.",
    "suicidewatch": "Suicide is the act of intentionally causing one's own death. Mental disorders (including depression, bipolar disorder, schizophrenia, personality disorders, anxiety disorders), physical disorders (such as chronic fatigue syndrome), and substance abuse (including alcoholism and the use of and withdrawal from benzodiazepines) are risk factors. Some suicides are impulsive acts due to stress (such as from financial or academic difficulties), relationship problems (such as breakups or divorces), or harassment and bullying. Those who have previously attempted suicide are at a higher risk for future attempts. Effective suicide prevention efforts include limiting access to methods of suicide such as firearms, drugs, and poisons; treating mental disorders and substance abuse; careful media reporting about suicide; and improving economic conditions.",
}