


def create_explanation_texts(plot, most_important_feats, importance_percentages, feature_vec, label, median, feature_description, threshold=.05):

    full_text = ""
    comparison_class = " a normal heartbeat signal" if label != "N" else " an atrial fibrillation signal"
    rel_median = median[median['label'] != label] 

    for feat_name, percent in zip(most_important_feats, importance_percentages):
        full_text += f"Rel: {round(percent, 3)}%     "
        other_median = rel_median[feat_name].to_numpy()[0]
        feature_comparison = feature_vec[feat_name].to_numpy()[0] - other_median
        if feature_comparison < other_median*(1-threshold):
            comp_string = " is lower than for"
        elif feature_comparison > other_median*(1+threshold):
            comp_string = " is greater than for"
        else:
            comp_string = " is the same as for"
        full_text += feature_description[feat_name] + comp_string + comparison_class + "\n"

    plot.text(0.005, 0.85, full_text, horizontalalignment='left', verticalalignment='center', transform=plot.transAxes, fontsize=15,
    va='top', wrap=True)
    
    """
        if feat_name == 'Number of p peaks missed':
            full_text += "The Number of p peaks missed is" + comp_string + comparison_class
        if feat_name == 'score1':
            full_text += "The proportion of R-R distances" + comp_string + comparison_class
        if feat_name == 'score2':
            full_text += "The proportion of R-R distances" + comp_string + comparison_class
        if feat_name == 'score3':
            full_text += "The proportion of R-R distances" + comp_string + comparison_class
        if feat_name == 'sd1':
            full_text += "The Short-term Heart rate variability rate" + comp_string + comparison_class
        if feat_name == 'sd2':
            full_text += "The Long-term Heart rate variability rate" + comp_string + comparison_class
        if feat_name == 'ratio':
            full_text += "The unpredictability of the RR" + comp_string + comparison_class
        if feat_name == 'beat_rate':
            full_text += "The Number of p peaks missed is" + comp_string + comparison_class
        if feat_name == 'dominant_freq':
            full_text += "The Number of p peaks missed is" + comp_string + comparison_class
        if feat_name == 'energy_percent_at_dominant_freq':
            full_text += "The Number of p peaks missed is" + comp_string + comparison_class
        if feat_name == 'mean1':
            full_text += "The Number of p peaks missed is" + comp_string + comparison_class
        if feat_name == 'std1':
            full_text += "The Number of p peaks missed is" + comp_string + comparison_class
        if feat_name == 'q2_1':
            full_text += "The Number of p peaks missed is" + comp_string + comparison_class
        if feat_name == 'iqr1':
            full_text += "The Number of p peaks missed is" + comp_string + comparison_class
        if feat_name == 'quartile_coeff_disp1':
            full_text += "The Number of p peaks missed is" + comp_string + comparison_class
        if feat_name == 'mean2':
            full_text += "The Number of p peaks missed is" + comp_string + comparison_class
        if feat_name == 'std2':
            full_text += "The Number of p peaks missed is" + comp_string + comparison_class
        if feat_name == 'q2_2':
            full_text += "The Number of p peaks missed is" + comp_string + comparison_class
        if feat_name == 'iqr2':
            full_text += "The Number of p peaks missed is" + comp_string + comparison_class
        if feat_name == 'quartile_coeff_disp2':
            full_text += "The Number of p peaks missed is" + comp_string + comparison_class
        if feat_name == 'mean3':
            full_text += "The Number of p peaks missed is" + comp_string + comparison_class
        if feat_name == 'std3':
            full_text += "The Number of p peaks missed is" + comp_string + comparison_class
        if feat_name == 'q2_3':
            full_text += "The Number of p peaks missed is" + comp_string + comparison_class
        if feat_name == 'iqr3':
            full_text += "The Number of p peaks missed is" + comp_string + comparison_class
        if feat_name == 'quartile_coeff_disp3':
            full_text += "The Number of p peaks missed is" + comp_string + comparison_class
        """