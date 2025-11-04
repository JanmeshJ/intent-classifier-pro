"""
Enhanced Dataset Generator - 7 Intent Classes
Generates 3,500 samples for production-grade training
"""

import pandas as pd
import random
from pathlib import Path

# 7 Intent Categories
INTENTS = {
    'product_inquiry': [
        "What are the specs of the {product}?",
        "How much does the {product} cost?",
        "Is the {product} available in {color}?",
        "Do you have {product} in stock?",
        "What colors does the {product} come in?",
        "Tell me about the {product} features",
        "What's the price of the {product}?",
        "Is the {product} waterproof?",
        "Does the {product} have a warranty?",
        "What's included with the {product}?",
    ],
    
    'technical_support': [
        "My {product} won't turn on",
        "The {product} keeps freezing",
        "I'm getting an error on my {product}",
        "The {feature} isn't working",
        "My {product} won't connect",
        "The screen is black on my {product}",
        "My {product} is running slow",
        "I can't update my {product}",
        "The {product} makes strange noises",
        "My {product} won't charge",
    ],
    
    'order_status': [
        "Where is my order?",
        "When will my {product} arrive?",
        "Can I track my shipment?",
        "Has my order been shipped?",
        "What's the status of my order?",
        "When was my package sent?",
        "How long until delivery?",
        "Can you check on my delivery?",
        "I haven't received my order",
        "Is my {product} on the way?",
    ],
    
    'returns_refunds': [
        "How do I return my {product}?",
        "Can I get a refund?",
        "I want to return this {product}",
        "What's your return policy?",
        "How long do I have to return?",
        "Can I exchange my {product}?",
        "I need to send back my {product}",
        "Where do I ship returns?",
        "Do you offer free returns?",
        "How do I start a return?",
    ],
    
    'account_billing': [
        "I was charged twice",
        "Can I update my payment method?",
        "How do I change billing address?",
        "I don't recognize this charge",
        "Can I get an invoice?",
        "How do I update my credit card?",
        "Why was I charged this amount?",
        "Can I change my payment plan?",
        "I need a receipt",
        "How do I cancel my subscription?",
    ],
    
    'shipping_delivery': [
        "Do you ship to {location}?",
        "What are the shipping costs?",
        "How long does shipping take?",
        "Do you offer express shipping?",
        "Can I change my shipping address?",
        "What shipping methods available?",
        "Is there free shipping?",
        "Can I pick up in store?",
        "Do you ship internationally?",
        "What's the fastest shipping?",
    ],
    
    'general_inquiry': [
        "What are your business hours?",
        "How can I contact support?",
        "Do you have a physical store?",
        "What's your phone number?",
        "Can I chat with someone?",
        "Where is your company?",
        "Do you have a mobile app?",
        "How do I create an account?",
        "Can I get your newsletter?",
        "What payment methods accepted?",
    ]
}

PRODUCTS = ['smartphone', 'laptop', 'tablet', 'smartwatch', 'headphones', 
           'camera', 'speaker', 'monitor', 'keyboard', 'mouse']
COLORS = ['black', 'white', 'silver', 'blue', 'red', 'gold']
FEATURES = ['bluetooth', 'wifi', 'GPS', 'NFC', '5G']
LOCATIONS = ['California', 'Canada', 'Europe', 'Australia']

def generate_sample(intent, template):
    text = template.format(
        product=random.choice(PRODUCTS),
        color=random.choice(COLORS),
        feature=random.choice(FEATURES),
        location=random.choice(LOCATIONS)
    )
    return {'text': text, 'intent': intent}

def main():
    print("=" * 60)
    print("GENERATING ENHANCED DATASET - 7 INTENT CLASSES")
    print("=" * 60)
    print()
    
    samples = []
    samples_per_class = 500  # 500 x 7 = 3,500 total
    
    for intent, templates in INTENTS.items():
        print(f"Generating {samples_per_class} samples for {intent}...")
        for _ in range(samples_per_class):
            template = random.choice(templates)
            sample = generate_sample(intent, template)
            samples.append(sample)
    
    random.shuffle(samples)
    df = pd.DataFrame(samples)
    
    # Create data directory
    Path('data').mkdir(exist_ok=True)
    
    # Save
    output_path = 'data/customer_intents.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Dataset generated: {len(df)} samples")
    print(f"✓ Saved to: {output_path}")
    print(f"\nIntent distribution:")
    print(df['intent'].value_counts())
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
