from nltk.translate.bleu_score import sentence_bleu
import logging

def calculate_bleu_score(reference, candidate):
    """
    Calculates the BLEU score between a reference translation and a candidate translation.
    """
    try:
        reference_list = [reference.split()]  # BLEU expects a list of reference sentences
        candidate_list = candidate.split()
        score = sentence_bleu(reference_list, candidate_list)
        logging.info(f"BLEU Score: {score}")
        return score
    except Exception as e:
        logging.error(f"Error calculating BLEU score: {e}")
        return 0.0  # Return 0.0 in case of error
