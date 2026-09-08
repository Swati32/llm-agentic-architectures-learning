"""Coarse-group taxonomy over Banking77's 77 intents.

Banking77 ships no official coarse categories, so this grouping is our own
methodological choice, authored by hand against the dataset's real label
strings (not invented from memory). It becomes ground truth for the
Coarse Accuracy and Hierarchy Consistency Rate metrics, so keep it in sync
with data.py's `INTENT_LABELS` if the dataset version ever changes.
"""

COARSE_GROUPS = {
    "Getting a Card": [
        "activate_my_card",
        "card_about_to_expire",
        "card_arrival",
        "card_delivery_estimate",
        "card_linking",
        "get_disposable_virtual_card",
        "get_physical_card",
        "getting_spare_card",
        "getting_virtual_card",
        "order_physical_card",
    ],
    "Card Problems": [
        "card_not_working",
        "card_swallowed",
        "change_pin",
        "compromised_card",
        "contactless_not_working",
        "disposable_card_limits",
        "lost_or_stolen_card",
        "lost_or_stolen_phone",
        "passcode_forgotten",
        "pin_blocked",
        "virtual_card_not_working",
    ],
    "Card Payments": [
        "card_payment_fee_charged",
        "card_payment_not_recognised",
        "card_payment_wrong_exchange_rate",
        "declined_card_payment",
        "extra_charge_on_statement",
        "pending_card_payment",
        "reverted_card_payment?",
        "transaction_charged_twice",
    ],
    "ATM & Cash Withdrawals": [
        "atm_support",
        "cash_withdrawal_charge",
        "cash_withdrawal_not_recognised",
        "declined_cash_withdrawal",
        "pending_cash_withdrawal",
        "wrong_amount_of_cash_received",
        "wrong_exchange_rate_for_cash_withdrawal",
    ],
    "Top-ups": [
        "automatic_top_up",
        "pending_top_up",
        "top_up_by_bank_transfer_charge",
        "top_up_by_card_charge",
        "top_up_by_cash_or_cheque",
        "top_up_failed",
        "top_up_limits",
        "top_up_reverted",
        "topping_up_by_card",
        "verify_top_up",
    ],
    "Transfers": [
        "balance_not_updated_after_bank_transfer",
        "balance_not_updated_after_cheque_or_cash_deposit",
        "beneficiary_not_allowed",
        "cancel_transfer",
        "declined_transfer",
        "direct_debit_payment_not_recognised",
        "failed_transfer",
        "pending_transfer",
        "receiving_money",
        "transfer_fee_charged",
        "transfer_into_account",
        "transfer_not_received_by_recipient",
        "transfer_timing",
    ],
    "Refunds & Disputes": [
        "Refund_not_showing_up",
        "request_refund",
    ],
    "Currency, Exchange & Compatibility": [
        "apple_pay_or_google_pay",
        "card_acceptance",
        "country_support",
        "exchange_charge",
        "exchange_rate",
        "exchange_via_app",
        "fiat_currency_support",
        "supported_cards_and_currencies",
        "visa_or_mastercard",
    ],
    "Identity Verification": [
        "unable_to_verify_identity",
        "verify_my_identity",
        "verify_source_of_funds",
        "why_verify_identity",
    ],
    "Account Administration": [
        "age_limit",
        "edit_personal_details",
        "terminate_account",
    ],
}

INTENT_TO_GROUP = {
    intent: group for group, intents in COARSE_GROUPS.items() for intent in intents
}

GROUP_NAMES = list(COARSE_GROUPS.keys())
NORMALIZED_TO_GROUP = {group.strip().lower(): group for group in GROUP_NAMES}


def resolve_group(raw_group: str) -> str | None:
    """Map a model's raw group text back to a canonical group name, or None
    if it doesn't match any of the 10 groups."""
    return NORMALIZED_TO_GROUP.get(raw_group.strip().lower())


def normalize_label(label: str) -> str:
    """Fold away dataset artifacts (casing, trailing punctuation) that carry
    no intent signal, so a technique isn't marked wrong for reproducing a
    quirk of the label string rather than misunderstanding the query."""
    return label.strip().rstrip("?").lower()


NORMALIZED_TO_INTENT = {normalize_label(intent): intent for intent in INTENT_TO_GROUP}


def resolve_intent(raw_label: str) -> str | None:
    """Map a model's raw label text back to a canonical intent, or None if
    it doesn't match any of the 77 labels even after normalization."""
    return NORMALIZED_TO_INTENT.get(normalize_label(raw_label))


def group_for_intent(intent: str) -> str | None:
    return INTENT_TO_GROUP.get(intent)
