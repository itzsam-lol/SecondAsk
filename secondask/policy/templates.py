"""DLT-registered message templates.

Under TRAI's TCCCPA framework, a business cannot send commercial SMS as free
text. Content must match a template registered on a DLT platform against a
registered header, and variable portions are restricted. That constraint is
usually treated as an integration detail and then violated in spirit by systems
that let a model write whatever it likes.

Here it is a type. A message is a ``template_id`` plus a dict of slot values.
There is no code path that sends free text. The language model fills declared
slots and nothing else, and ``render`` refuses if a slot is missing, unexpected,
over-long, or contains a character that would break out of the template.

Templates are also classified transactional vs promotional, because the rules
differ: promotional content requires consent and is subject to DND scrubbing,
transactional content is not. Misclassifying an offer as transactional to dodge
consent is the most common real-world abuse, so classification lives with the
template, not with the caller.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

MAX_SLOT_LEN = 30
MAX_SMS_LEN = 320

# Characters that must never appear in a slot value. Newlines would let a slot
# forge additional message lines; braces would let it inject another slot.
_FORBIDDEN_SLOT = re.compile(r"[\r\n{}<>]")


class TemplateError(ValueError):
    pass


@dataclass(frozen=True)
class Template:
    template_id: str
    dlt_id: str
    header: str
    body: str
    slots: tuple[str, ...]
    promotional: bool
    language: str = "en"

    def render(self, values: dict[str, Any]) -> str:
        """Fill the template. Refuses anything that is not exactly right.

        Fails closed on every one of: missing slot, unknown slot, over-long
        value, or a value containing control or structural characters.
        """
        missing = [s for s in self.slots if s not in values]
        if missing:
            raise TemplateError(f"template {self.template_id}: missing slots {missing}")
        extra = [k for k in values if k not in self.slots]
        if extra:
            raise TemplateError(f"template {self.template_id}: unexpected slots {extra}")

        clean: dict[str, str] = {}
        for slot in self.slots:
            raw = values[slot]
            text = raw if isinstance(raw, str) else str(raw)
            if len(text) > MAX_SLOT_LEN:
                raise TemplateError(
                    f"template {self.template_id}: slot '{slot}' is {len(text)} chars, max {MAX_SLOT_LEN}"
                )
            if _FORBIDDEN_SLOT.search(text):
                raise TemplateError(
                    f"template {self.template_id}: slot '{slot}' contains a forbidden character"
                )
            clean[slot] = text

        rendered = self.body
        for slot, text in clean.items():
            rendered = rendered.replace("{" + slot + "}", text)
        if "{" in rendered or "}" in rendered:
            raise TemplateError(f"template {self.template_id}: unresolved placeholder after render")
        rendered = f"{rendered}\n-{self.header}"
        if len(rendered) > MAX_SMS_LEN:
            raise TemplateError(f"template {self.template_id}: rendered body exceeds {MAX_SMS_LEN} chars")
        return rendered


TEMPLATES: dict[str, Template] = {}


def _register(template: Template) -> None:
    TEMPLATES[template.template_id] = template


_register(
    Template(
        template_id="RETRY_LINK_EN",
        dlt_id="1207161234567890123",
        header="MRCHNT",
        body="Your payment of Rs {amount} to {merchant} did not go through. Complete it here: {link}",
        slots=("amount", "merchant", "link"),
        promotional=False,
    )
)
_register(
    Template(
        template_id="RETRY_LINK_HI",
        dlt_id="1207161234567890124",
        header="MRCHNT",
        body="{merchant} ko aapka Rs {amount} ka payment pura nahi hua. Yahan complete karein: {link}",
        slots=("amount", "merchant", "link"),
        promotional=False,
        language="hi",
    )
)
_register(
    Template(
        template_id="UPDATE_CARD_EN",
        dlt_id="1207161234567890125",
        header="MRCHNT",
        body="Your saved card for {merchant} is no longer usable. Update it here to avoid interruption: {link}",
        slots=("merchant", "link"),
        promotional=False,
    )
)
_register(
    Template(
        template_id="UPDATE_CARD_HI",
        dlt_id="1207161234567890126",
        header="MRCHNT",
        body="{merchant} ke liye aapka card ab kaam nahi kar raha. Yahan update karein: {link}",
        slots=("merchant", "link"),
        promotional=False,
        language="hi",
    )
)
_register(
    Template(
        # RBI requires a pre-debit notification at least 24 hours before an
        # e-mandate debit, stating amount and date so the customer can cancel.
        template_id="PREDEBIT_NOTICE_EN",
        dlt_id="1207161234567890127",
        header="MRCHNT",
        body="Notice: Rs {amount} will be debited for {merchant} on {date}. To cancel, visit {link}",
        slots=("amount", "merchant", "date", "link"),
        promotional=False,
    )
)
_register(
    Template(
        template_id="PREDEBIT_NOTICE_HI",
        dlt_id="1207161234567890128",
        header="MRCHNT",
        body="Suchna: {merchant} ke liye Rs {amount} {date} ko debit hoga. Cancel karne ke liye: {link}",
        slots=("amount", "merchant", "date", "link"),
        promotional=False,
        language="hi",
    )
)
_register(
    Template(
        # Promotional: it offers a discount, so it needs consent and DND scrubbing.
        template_id="INCENTIVE_EN",
        dlt_id="1207161234567890129",
        header="MRCHNT",
        body="Complete your {merchant} order today and save Rs {discount}. Pay here: {link}",
        slots=("merchant", "discount", "link"),
        promotional=True,
    )
)
_register(
    Template(
        template_id="INVOICE_DUE_EN",
        dlt_id="1207161234567890130",
        header="MRCHNT",
        body="Invoice {invoice} for Rs {amount} from {merchant} is overdue since {date}. Settle: {link}",
        slots=("invoice", "amount", "merchant", "date", "link"),
        promotional=False,
    )
)
_register(
    Template(
        # The MSMED angle. Stating a statutory consequence is a factual notice,
        # not a threat, and it is materially more effective than "please pay".
        template_id="INVOICE_MSME_EN",
        dlt_id="1207161234567890131",
        header="MRCHNT",
        body=(
            "Invoice {invoice} (Rs {amount}) is {days} days past the MSMED Act limit. "
            "Interest accrues and your deduction is at risk u/s 43B(h). Settle: {link}"
        ),
        slots=("invoice", "amount", "days", "link"),
        promotional=False,
    )
)


def get_template(template_id: str) -> Template:
    template = TEMPLATES.get(template_id)
    if template is None:
        raise TemplateError(f"unregistered template id: {template_id!r}")
    return template


def template_for(kind: str, language: str) -> str:
    """Pick a registered template for an action kind and language.

    Falls back to English rather than inventing a template, because an
    unregistered send is a compliance breach and a slightly wrong language is
    merely suboptimal.
    """
    base = {
        "payment_link": "RETRY_LINK",
        "update_instrument": "UPDATE_CARD",
        "prenotify": "PREDEBIT_NOTICE",
    }.get(kind)
    if base is None:
        return {"incentive": "INCENTIVE_EN", "invoice": "INVOICE_DUE_EN"}.get(kind, "RETRY_LINK_EN")
    suffix = "HI" if language in ("hi", "hinglish") else "EN"
    candidate = f"{base}_{suffix}"
    return candidate if candidate in TEMPLATES else f"{base}_EN"
