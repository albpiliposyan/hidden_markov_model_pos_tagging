"""Tag custom Armenian text with trained HMM model."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hmm import HiddenMarkovModel


def tokenize_armenian(text):
    """Simple tokenizer for Armenian text."""
    # Split by whitespace and punctuation
    import re
    # Keep punctuation as separate tokens
    tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    return [token for token in tokens if token.strip()]


def tag_text(text, model_path='models/hmm_armenian_pos.pkl'):
    """Tag Armenian text with POS tags."""
    
    # Load trained model
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run src/main.py first to train the model.")
        return
    
    hmm = HiddenMarkovModel.load(model_path)
    
    # Split text into lines (treating each line as a sentence)
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    
    print("\n" + "="*70)
    print("POS TAGGING RESULTS")
    print("="*70)
    
    for line_num, line in enumerate(lines, 1):
        print(f"\nLine {line_num}:")
        print(f"Text: {line}")
        print()
        
        # Tokenize
        words = tokenize_armenian(line)
        
        if not words:
            print("  (empty line)")
            continue
        
        # Predict tags
        tags = hmm.predict(words)
        
        # Display results
        print(f"{'№':<4} {'Word':<20} {'POS Tag':<10} {'Known/Unknown':<15}")
        print("-" * 50)
        for i, (word, tag) in enumerate(zip(words, tags), 1):
            known_status = 'Known' if word.lower() in hmm.word_to_idx else 'Unknown'
            print(f"{i:<4} {word:<20} {tag:<10} {known_status:<15}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # armenian_text = """
    # Ես իմ անուշ Հայաստանի արևահամ բարն եմ սիրում,
    # Մեր հին սազի ողբանվագ, լացակումած լարն եմ սիրում,
    # Արնանման ծաղիկների ու վարդերի բույրը վառման,
    # Ու նաիրյան աղջիկների հեզաճկուն պարն եմ սիրում։
    # """

    # armenian_text = """
    # Տիեզերքում աստվածային մի ճամփորդ է իմ հոգին.
    # Երկրից անցվոր, երկրի փառքին անհաղորդ է իմ հոգին.
    # Հեռացել է ու վերացել մինչ աստղերը հեռավոր,
    # Վար մնացած մարդու համար արդեն խորթ է իմ հոգին։
    # """

    # armenian_text = """
    # Աբու–Լալա Մահարին, Հռչակավոր բանաստեղծը Բաղդադի, Տասնյակ տարիներ ապրեց Խալիֆաների հոյակապ քաղաքում, Ապրեց փառքի և վայելքի մեջ, Հզորների և մեծատուների հետ սեղան նստեց, Գիտունների և իմաստունների հետ վեճի մտավ, Սիրեց և փորձեց ընկերներին, Եղավ ուրիշ – ուրիշ ազգերի հայրենիքներում, Տեսավ և դիտեց մարդկանց և օրենքները: Եվ նրա խորաթափանց ոգին ճանաչեց մարդուն, Ճանաչեց և խորագին ատեց մարդուն Եվ նրա օրենքները:
    # Եվ որովհետև չուներ կին և երեխաներ, Բոլոր իր հարստությունը բաժանեց աղքատներին, Առավ իր ուղտերի փոքրիկ քարավանը` պաշարով ու պարենով, Եվ մի գիշեր, երբ Բաղդադը քուն էր մտել Տիգրիսի նոճիածածկ ափերի վրա, – Գաղտնի հեռացավ քաղաքից…
    # """

    armenian_text = """
    Թողնել ընտանիքը անտեր, երկար ու ձիգ տարիներ թափառել աշխարհի մի ծայրից դեպի մյուսը, արծաթ որսալու համար չխնայել ամեն տեսակ անազնիվ միջոցներ այդ խաչագողի գործն է։
    Խաչագողը ունի իր արհեստին վերաբերյալ բոլոր հմտությունները։ Այլևայլ երկրներում թափառելու համար նա գիտե զանազան ազգերի լեզուներ, ծանոթ է նրանց սովորություններին և ինչ ազգի մեջ որ մտնում է, խոսում է այն քան վարժ, որ դժվար է որոշել, թե նա այն ազգին չէ պատկանում։ Նա իր ընկերների հետ խոսում է մի առանձին լեզվով, որը ոչ ոք հասկանալ կարող չէ, եթե խաչագողների հասարակությանը չէ պատկանում։ Դա մի խորհրդավոր, պայմանական լեզու է. դա ավազակների արգոն է։
    """

    tag_text(armenian_text)
