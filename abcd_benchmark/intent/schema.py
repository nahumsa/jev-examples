"""Typed intent vocabulary and output schema with preserved option descriptions."""

from enum import Enum

from pydantic import BaseModel, ConfigDict
from pydantic_ai import UseEnumMemberDocstrings


class Intent(UseEnumMemberDocstrings, str, Enum):
    recover_username = 'recover_username'
    """Account access: recover a forgotten username."""
    recover_password = 'recover_password'
    """Account access: recover or reset a forgotten password."""
    reset_2fa = 'reset_2fa'
    """Account access: reset two-factor authentication."""
    status_service_added = 'status_service_added'
    """Account management: an unexpected service was added to the account."""
    status_service_removed = 'status_service_removed'
    """Account management: a service was unexpectedly removed from the account."""
    status_shipping_question = 'status_shipping_question'
    """Account management: ask whether the account allows free international shipping."""
    status_credit_missing = 'status_credit_missing'
    """Account management: expected account credit is missing."""
    manage_change_address = 'manage_change_address'
    """Account management: change the address on the account."""
    manage_change_name = 'manage_change_name'
    """Account management: change the name on the account."""
    manage_change_phone = 'manage_change_phone'
    """Account management: change the phone number on the account."""
    manage_payment_method = 'manage_payment_method'
    """Account management: change the account's payment method."""
    status_mystery_fee = 'status_mystery_fee'
    """Order issue: ask about an unexplained fee on an order."""
    status_delivery_time = 'status_delivery_time'
    """Order issue: ask when an order will be delivered."""
    status_payment_method = 'status_payment_method'
    """Order issue: ask which payment method was used on an order."""
    status_quantity = 'status_quantity'
    """Order issue: question the number of products in an order."""
    manage_upgrade = 'manage_upgrade'
    """Order issue: upgrade an existing order."""
    manage_downgrade = 'manage_downgrade'
    """Order issue: downgrade an existing order."""
    manage_create = 'manage_create'
    """Order issue: create a new order or purchase."""
    manage_cancel = 'manage_cancel'
    """Order issue: cancel an existing order."""
    refund_initiate = 'refund_initiate'
    """Product defect: initiate a new refund."""
    refund_update = 'refund_update'
    """Product defect: add an item to or update an existing refund."""
    refund_status = 'refund_status'
    """Product defect: ask about the progress of an existing refund."""
    return_stain = 'return_stain'
    """Product defect: return a stained or dirty item."""
    return_color = 'return_color'
    """Product defect: return an item because its color is wrong."""
    return_size = 'return_size'
    """Product defect: return an item because its size or fit is wrong."""
    bad_price_competitor = 'bad_price_competitor'
    """Purchase dispute: a competitor offers a lower price."""
    bad_price_yesterday = 'bad_price_yesterday'
    """Purchase dispute: the store's price was lower yesterday."""
    out_of_stock_general = 'out_of_stock_general'
    """Purchase dispute: complain generally about products being out of stock."""
    out_of_stock_one_item = 'out_of_stock_one_item'
    """Purchase dispute: a particular desired item is out of stock."""
    promo_code_invalid = 'promo_code_invalid'
    """Purchase dispute: a promotional code is invalid or does not work, not explicitly expired."""
    promo_code_out_of_date = 'promo_code_out_of_date'
    """Purchase dispute: a promotional code has expired."""
    mistimed_billing_already_returned = 'mistimed_billing_already_returned'
    """Purchase dispute: charged for an item already returned."""
    mistimed_billing_never_bought = 'mistimed_billing_never_bought'
    """Purchase dispute: charged for an item the customer never bought."""
    status = 'status'
    """Shipping issue: check shipment status or tracking."""
    manage = 'manage'
    """Shipping issue: change or manage shipping arrangements."""
    missing = 'missing'
    """Shipping issue: a shipment or package is missing."""
    cost = 'cost'
    """Shipping issue: question or dispute shipping costs."""
    boots = 'boots'
    """Single-item query: ask general product information about boots."""
    shirt = 'shirt'
    """Single-item query: ask general product information about shirts."""
    jeans = 'jeans'
    """Single-item query: ask general product information about jeans."""
    jacket = 'jacket'
    """Single-item query: ask general product information about jackets."""
    pricing = 'pricing'
    """Storewide query: prices of gift wrapping, name stitching, overnight shipping, or eligibility for free shipping."""
    membership = 'membership'
    """Storewide query: ask about membership levels or benefits."""
    timing = 'timing'
    """Storewide query: collection release dates, store opening times, annual sale dates, or promo-code expiration dates."""
    policy = 'policy'
    """Storewide query: ask about general store policies."""
    status_active = 'status_active'
    """Subscription inquiry: ask whether a subscription is active."""
    status_due_amount = 'status_due_amount'
    """Subscription inquiry: ask how much is owed on a subscription bill."""
    status_due_date = 'status_due_date'
    """Subscription inquiry: ask when a subscription bill is due."""
    manage_pay_bill = 'manage_pay_bill'
    """Subscription inquiry: pay a subscription bill."""
    manage_extension = 'manage_extension'
    """Subscription inquiry: request more time to pay a subscription bill."""
    manage_dispute_bill = 'manage_dispute_bill'
    """Subscription inquiry: dispute a subscription bill."""
    credit_card = 'credit_card'
    """Website troubleshooting: credit-card entry or checkout is not working."""
    shopping_cart = 'shopping_cart'
    """Website troubleshooting: the shopping cart is not working."""
    search_results = 'search_results'
    """Website troubleshooting: search results are not working correctly."""
    slow_speed = 'slow_speed'
    """Website troubleshooting: the site is slow."""


class IntentPrediction(BaseModel):
    """Identify the customer's primary ABCD support intent."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    intent: Intent
    """Which ABCD intent best explains the customer's primary request in `dialogue`?
    Use only the observed dialogue. Distinguish the original request from routine
    identity verification, agent procedures, and later follow-up requests.
    """
