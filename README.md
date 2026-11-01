# E-Commerce Customer Segmentation — RFM Analysis

**Veda Technology internship project.** Analysis uses the supplied `OnlineRetail.xlsx` (UCI Online Retail style) as provided; there are no fabricated rows.

## Dataset and results
- Source: one sheet, 541,909 rows × 8 columns; 2010-12-01 through 2011-12-09.
- Source has 135,080 missing CustomerID cells and 5,268 duplicate rows. Cleaning removes sequentially: duplicates (5,268), missing CustomerID (135,037 after deduplication), cancelled invoices (8,872), and non-positive quantity/unit price (40). No invalid dates were observed after customer filtering.
- Retained: 392,692 purchase lines; 4,338 customers. Reference date: 2011-12-10.
- RFM: Recency is days from reference date to last invoice date; Frequency counts distinct invoices; Monetary sums Quantity × UnitPrice. Quintile scores are 1–5, with the Recency direction reversed; the concatenated RFM_Score is a string such as 545.
- K-Means uses log1p features and StandardScaler. k was selected by highest silhouette score among 2–8, checked against elbow curve. Best k=2 (silhouette 0.433). The observed segmentation supports two groups, not four conventional labels; names are based on observed profile.

## Contents
- `RFM_Customer_Segmentation.ipynb`: beginner-oriented notebook; run from this folder.
- `rfm_project.py`: full reproducible analysis pipeline.
- `OnlineRetail.xlsx`: supplied source workbook.
- `Cleaned_Online_Retail.csv`: cleaned transaction-level file.
- `RFM_Customer_Scores.csv`: one row per customer with RFM metrics, quintile scores, cluster and segment.
- `Customer_Clusters.csv`: customer cluster assignments and segment names (includes RFM metrics).
- `Cluster_Profile.csv`: actual cluster averages, medians, revenue and counts.
- `Business_Recommendation.pdf`: one-page, data-supported marketing recommendation.
- `RFM_Cluster_Visualization.png`, `segment_distribution.png`, `rfm_comparison.png`, `revenue_by_segment.png`, `elbow_method.png`: charts.
- `analysis_summary.json`: source inspection, cleaning counts and model metrics.

## Run locally
Python 3.10+ recommended. Install `pip install -r requirements.txt`, then run `python rfm_project.py`. It writes outputs to the same directory. The notebook runs the script in its next cell.

## Dashboard instructions (Power BI or Tableau)
Import `Customer_Clusters.csv` and `Cleaned_Online_Retail.csv`; relate customer IDs (one customer to many line items). Add KPI cards for distinct customer count and revenue; charts for customer count/revenue by Segment; compare R, F and M metrics by Segment; add Country slicer from cleaned transaction data. Check totals against the CSV exports. A native PBIX/TWB file is not included.

## Interpretation cautions
The records are historical (2010–2011); this does not indicate current engagement. Monetary is historical line sales rather than margin/profit. Unknown IDs are removed from customer-level analysis; transaction-level unknowns cannot be assigned. The labels are descriptive summaries, not verified marketing personas. Use a holdout test before investing in campaigns.
