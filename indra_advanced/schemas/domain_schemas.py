"""
Standard Pre-built Domain Schemas for Indra-Advanced:
1. Invoice & Purchase Order
2. Financial Reports & Balance Sheets
3. Clinical / Medical Lab Test Reports
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from indra_advanced.schemas.base_schema import ExtractionSchema


# -------------------------------------------------------------
# 1. Invoice Document Schema
# -------------------------------------------------------------
class InvoiceLineItem(BaseModel):
    item_description: str = Field(description="Description of product, service, or line item")
    quantity: Optional[str] = Field(default=None, description="Quantity or unit count")
    unit_price: Optional[str] = Field(default=None, description="Price per individual unit")
    item_total: Optional[str] = Field(default=None, description="Extended line item total cost")


class InvoiceDocument(BaseModel):
    vendor_name: Optional[str] = Field(default=None, description="Name of seller or vendor issuing the invoice")
    invoice_number: Optional[str] = Field(default=None, description="Unique invoice identifier or reference code")
    invoice_date: Optional[str] = Field(default=None, description="Date the invoice was issued")
    due_date: Optional[str] = Field(default=None, description="Payment due date")
    customer_name: Optional[str] = Field(default=None, description="Name of buyer or customer billed")
    subtotal: Optional[str] = Field(default=None, description="Pre-tax subtotal amount")
    tax_amount: Optional[str] = Field(default=None, description="Total sales tax or VAT charged")
    total_amount: Optional[str] = Field(default=None, description="Final gross total amount payable")
    currency: Optional[str] = Field(default=None, description="Currency symbol or code (USD, EUR, GBP, INR, etc.)")


INVOICE_SCHEMA = ExtractionSchema(
    name="invoice_document",
    description="Extracts structured invoice and purchase order fields",
    entity_labels=[
        "vendor_name",
        "invoice_number",
        "invoice_date",
        "due_date",
        "customer_name",
        "subtotal",
        "tax_amount",
        "total_amount",
        "item_description",
        "quantity",
        "unit_price",
        "item_total",
    ],
    label_descriptions={
        "vendor_name": "Supplier, company, or vendor name",
        "invoice_number": "Invoice ID or bill number",
        "invoice_date": "Date of the invoice",
        "due_date": "Due date for payment",
        "customer_name": "Client or bill-to party name",
        "subtotal": "Net subtotal before tax",
        "tax_amount": "Tax amount or VAT",
        "total_amount": "Total amount due",
        "item_description": "Purchased item or service description",
        "quantity": "Quantity ordered or billed",
        "unit_price": "Rate or price per unit",
        "item_total": "Total line amount for item",
    },
)


# -------------------------------------------------------------
# 2. Financial Statements Schema
# -------------------------------------------------------------
class FinancialStatement(BaseModel):
    company_name: Optional[str] = Field(default=None, description="Company reporting the financials")
    reporting_period: Optional[str] = Field(default=None, description="Fiscal year or quarter, e.g. Q3 2024")
    revenue: Optional[str] = Field(default=None, description="Total revenue or gross sales")
    cost_of_revenue: Optional[str] = Field(default=None, description="Cost of goods sold (COGS)")
    gross_profit: Optional[str] = Field(default=None, description="Gross profit")
    operating_expenses: Optional[str] = Field(default=None, description="Operating expenses (R&D, SG&A)")
    operating_income: Optional[str] = Field(default=None, description="Operating income or EBIT")
    net_income: Optional[str] = Field(default=None, description="Net income or profit after tax")
    total_assets: Optional[str] = Field(default=None, description="Total balance sheet assets")
    total_liabilities: Optional[str] = Field(default=None, description="Total balance sheet liabilities")
    earnings_per_share: Optional[str] = Field(default=None, description="Diluted or basic EPS")


FINANCIAL_SCHEMA = ExtractionSchema(
    name="financial_statement",
    description="Extracts financial statement metrics from corporate earnings reports",
    entity_labels=[
        "company_name",
        "reporting_period",
        "revenue",
        "cost_of_revenue",
        "gross_profit",
        "operating_expenses",
        "operating_income",
        "net_income",
        "total_assets",
        "total_liabilities",
        "earnings_per_share",
    ],
)


# -------------------------------------------------------------
# 3. Medical / Clinical Lab Report Schema
# -------------------------------------------------------------
class MedicalLabReport(BaseModel):
    patient_name: Optional[str] = Field(default=None, description="Patient's full name")
    sample_id: Optional[str] = Field(default=None, description="Specimen or sample accession number")
    test_date: Optional[str] = Field(default=None, description="Collection or testing date")
    test_name: Optional[str] = Field(default=None, description="Name of laboratory test (e.g. Hemoglobin, Glucose)")
    result_value: Optional[str] = Field(default=None, description="Measured quantitative value")
    units: Optional[str] = Field(default=None, description="Measurement units (e.g. mg/dL, mmol/L)")
    reference_range: Optional[str] = Field(default=None, description="Normal biological reference interval")
    abnormal_flag: Optional[str] = Field(default=None, description="High, Low, or Abnormal indicator flag")


MEDICAL_LAB_SCHEMA = ExtractionSchema(
    name="medical_lab_report",
    description="Extracts laboratory diagnostic test measurements and reference bounds",
    entity_labels=[
        "patient_name",
        "sample_id",
        "test_date",
        "test_name",
        "result_value",
        "units",
        "reference_range",
        "abnormal_flag",
    ],
)
