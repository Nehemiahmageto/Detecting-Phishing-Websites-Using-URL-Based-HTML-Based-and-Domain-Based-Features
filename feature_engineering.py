"""
Advanced Feature Engineering for Phishing Detection
This script creates new features from existing ones to improve model performance
"""

import pandas as pd
import numpy as np


def create_url_complexity_score(df):
    """
    Creates a URL complexity score based on special characters and structure
    Higher scores indicate more complex/suspicious URLs
    """
    df['url_complexity_score'] = (
        df['nb_dots'] * 0.3 +
        df['nb_hyphens'] * 0.2 +
        df['nb_at'] * 2.0 +  # @ symbol is highly suspicious
        df['nb_underscore'] * 0.15 +
        df['nb_percent'] * 0.5 +
        df['nb_and'] * 0.2 +
        df['nb_eq'] * 0.2 +
        (df['length_url'] / 100) * 0.5  # Normalize URL length
    )
    return df


def create_domain_trust_score(df):
    """
    Creates a domain trust score based on domain characteristics
    Higher scores indicate more trustworthy domains
    """
    # Normalize domain age (older = more trustworthy)
    max_age = df['domain_age'].max()
    normalized_age = df['domain_age'] / max_age if max_age > 0 else 0
    
    # Normalize web traffic (more traffic = more trustworthy)
    max_traffic = df['web_traffic'].max()
    normalized_traffic = df['web_traffic'] / max_traffic if max_traffic > 0 else 0
    
    df['domain_trust_score'] = (
        normalized_age * 0.35 +
        (df['domain_registration_length'] / 365) * 0.25 +  # Years registered
        normalized_traffic * 0.2 +
        df['dns_record'] * 0.1 +
        df['google_index'] * 0.1
    )
    return df


def create_suspicious_char_ratio(df):
    """
    Ratio of suspicious characters to total URL length
    """
    suspicious_chars = (
        df['nb_at'] +
        df['nb_percent'] +
        df['nb_and'] +
        df['nb_eq'] +
        df['nb_tilde'] +
        df['nb_dollar']
    )
    
    df['suspicious_char_ratio'] = suspicious_chars / (df['length_url'] + 1)  # +1 to avoid division by zero
    return df


def create_brand_spoofing_indicator(df):
    """
    Combines brand-related features to detect potential spoofing attempts
    """
    df['brand_spoofing_indicator'] = (
        df['domain_in_brand'] * 2 +  # Domain mimicking brand
        df['brand_in_subdomain'] * 1.5 +  # Brand name in subdomain
        df['brand_in_path'] +  # Brand name in path
        df['phish_hints']  # Phishing keywords
    )
    return df


def create_redirect_danger_score(df):
    """
    Measures potential danger from redirections
    """
    df['redirect_danger_score'] = (
        df['nb_redirection'] * 0.5 +
        df['nb_external_redirection'] * 2 +  # External redirects more dangerous
        df['ratio_extRedirection'] * 2
    )
    return df


def create_url_structure_anomaly(df):
    """
    Detects structural anomalies in URLs
    """
    df['url_structure_anomaly'] = (
        df['tld_in_path'] * 2 +  # TLD appearing in path is very suspicious
        df['tld_in_subdomain'] * 1.5 +
        df['abnormal_subdomain'] * 2 +
        df['prefix_suffix'] * 1.5 +  # Hyphens in domain name
        df['http_in_path'] +  # HTTP/HTTPS in path
        df['shortening_service'] * 1.5
    )
    return df


def create_host_path_length_ratio(df):
    """
    Ratio of hostname length to total URL length
    Phishing URLs often have long paths relative to short hostnames
    """
    df['host_path_length_ratio'] = df['length_hostname'] / (df['length_url'] + 1)
    return df


def create_word_consistency_score(df):
    """
    Measures consistency in word lengths (randomness indicator)
    Higher variation suggests more random/generated URLs
    """
    # Calculate variance in word lengths
    avg_word = df['avg_words_raw']
    longest_word = df['longest_words_raw']
    shortest_word = df['shortest_words_raw']
    
    # Standard deviation approximation
    df['word_length_variance'] = ((longest_word - shortest_word) / (avg_word + 1))
    return df


def create_hyperlink_risk_score(df):
    """
    Assesses risk based on hyperlink patterns
    """
    df['hyperlink_risk_score'] = (
        df['ratio_extHyperlinks'] * 2 +  # External links more risky
        df['ratio_nullHyperlinks'] * 1.5 +
        (1 - df['ratio_intHyperlinks']) +  # Fewer internal links = more risky
        df['ratio_extErrors'] * 2
    )
    return df


def create_security_features_aggregate(df):
    """
    Aggregates various security-related features
    """
    df['security_risk_score'] = (
        (1 - df['https_token']) * 2 +  # No HTTPS is risky
        df['ip'] * 2 +  # IP address instead of domain
        df['port'] * 1.5 +  # Non-standard port
        df['suspecious_tld'] * 2 +  # Suspicious TLD
        (1 - df['whois_registered_domain']) * 1.5  # Not registered in WHOIS
    )
    return df


def create_digit_concentration(df):
    """
    Measures concentration of digits in different URL parts
    """
    # Higher ratio in hostname is more suspicious
    df['digit_concentration_host'] = df['ratio_digits_host'] * 2
    
    # Compare URL vs host digit ratios
    df['digit_ratio_diff'] = abs(df['ratio_digits_url'] - df['ratio_digits_host'])
    return df


def create_interaction_features(df):
    """
    Creates interaction features between important variables
    """
    # Short domain but long URL is suspicious
    df['short_domain_long_url'] = (df['length_url'] > 75) & (df['length_hostname'] < 20)
    df['short_domain_long_url'] = df['short_domain_long_url'].astype(int)
    
    # Many subdomains with suspicious TLD
    df['many_subdomains_suspicious_tld'] = (df['nb_subdomains'] * df['suspecious_tld'])
    
    # Young domain with low traffic
    df['young_domain_low_traffic'] = ((df['domain_age'] < 365) & (df['web_traffic'] < 1000)).astype(int)
    
    return df


def engineer_all_features(df):
    """
    Applies all feature engineering functions to the dataframe
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe with original features
        
    Returns:
    --------
    df : pandas DataFrame
        Dataframe with all engineered features added
    """
    print("Starting feature engineering...")
    
    print("Creating URL complexity score...")
    df = create_url_complexity_score(df)
    
    print("Creating domain trust score...")
    df = create_domain_trust_score(df)
    
    print("Creating suspicious character ratio...")
    df = create_suspicious_char_ratio(df)
    
    print("Creating brand spoofing indicator...")
    df = create_brand_spoofing_indicator(df)
    
    print("Creating redirect danger score...")
    df = create_redirect_danger_score(df)
    
    print("Creating URL structure anomaly score...")
    df = create_url_structure_anomaly(df)
    
    print("Creating host/path length ratio...")
    df = create_host_path_length_ratio(df)
    
    print("Creating word consistency score...")
    df = create_word_consistency_score(df)
    
    print("Creating hyperlink risk score...")
    df = create_hyperlink_risk_score(df)
    
    print("Creating security risk score...")
    df = create_security_features_aggregate(df)
    
    print("Creating digit concentration features...")
    df = create_digit_concentration(df)
    
    print("Creating interaction features...")
    df = create_interaction_features(df)
    
    print(f"Feature engineering complete! Added {len([col for col in df.columns if '_score' in col or '_ratio' in col or '_indicator' in col])} new features.")
    
    return df


# Example usage
if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv('dataset_phishing.csv')
    
    # Apply feature engineering
    df_engineered = engineer_all_features(df)
    
    # Display new features
    new_features = [col for col in df_engineered.columns if col not in pd.read_csv('dataset_phishing.csv').columns]
    print(f"\nNew features created: {new_features}")
    
    # Save enhanced dataset
    df_engineered.to_csv('dataset_phishing_engineered.csv', index=False)
    print("\nEnhanced dataset saved as 'dataset_phishing_engineered.csv'")
