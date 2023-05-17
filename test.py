from text_infiller import TextInfiller


infiller = TextInfiller(model_path='model/anli_anlg_positive_google/t5-v1_1-xl', deberta_model_path='model/anli_classifier_aug_xl_high_prob_2x/', deberta_filter=True)
test = {"story_id": "ffd2ab27-5d04-4cd7-b079-ae1d91a2bb32-1", \
    "obs1": "Jimmy was a notorious freestyle rapper in his community.", \
        "obs2": "They cheered for him to win when he finished."}
test = {'obs1': "Rick grew up in a troubled household. He never found good support in family, and turned to gangs.", 'obs2': "The incident caused him to turn a new leaf. He is happy now."}

print(*infiller.predict(test['obs1'], test['obs2'], deberta_filter=True, forbidden_output='')[1], sep='\n')