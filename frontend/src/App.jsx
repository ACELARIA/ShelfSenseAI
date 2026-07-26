import { useState } from "react";
import UploadForm from "./components/UploadForm";
import "./App.css";

function App() {
    const [result, setResult] = useState(null);

    const productInfo = result?.product_info || {};
    const aiAnalysis = result?.analysis || {};
    const recommendations = result?.insights || {};
    const ocrText = result?.ocr?.cleaned_text || [];
    const reconstructedText = result?.ocr?.reconstructed_text || "";
    const ocrConfidence = result?.ocr?.confidence || [];
    const rawText = result?.ocr?.raw_text || [];
    const averageConfidence = ocrConfidence.length
        ? Math.round((ocrConfidence.reduce((sum, value) => sum + value, 0) / ocrConfidence.length) * 100)
        : 0;

    const parseOcrValue = (pattern) => {
        const text = (ocrText || []).join(" ");
        const match = text.match(pattern);
        return match ? match[1]?.trim() : "—";
    };

    const batchNumber = parseOcrValue(/batch\s*[:#-]?\s*([A-Za-z0-9-]+)/i);
    const manufacturingDate = parseOcrValue(/mfg\s*[:#-]?\s*([0-9]{1,2}[/-][0-9]{4}|[0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2}|[0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{4})/i);
    const mrp = parseOcrValue(/(?:mrp|price|rs|₹)\s*[:=]?\s*([0-9]+(?:\.[0-9]{1,2})?)/i);

    const completenessScore = Math.round(
        ((productInfo.product_name ? 1 : 0) +
            (productInfo.manufacturer ? 1 : 0) +
            (productInfo.category ? 1 : 0) +
            (productInfo.expiry_date ? 1 : 0) +
            (productInfo.ingredients?.length ? 1 : 0)) / 5 * 100
    );

    const badgeTone = (value) => {
        if (!value || value === "—") return "badge neutral";
        if (["success", "identified", "available", "valid", "low", "compliant", "suitable"].includes(String(value).toLowerCase())) {
            return "badge success";
        }
        if (["warning", "medium", "normal", "monitor", "pending"].includes(String(value).toLowerCase())) {
            return "badge warning";
        }
        if (["critical", "expired", "high", "danger", "false", "not identified"].includes(String(value).toLowerCase())) {
            return "badge danger";
        }
        return "badge neutral";
    };

    const renderField = (label, value, tone = "neutral") => (
        <div className="field-row">
            <span className="field-label">{label}</span>
            <span className={`badge ${tone}`}>{value || "—"}</span>
        </div>
    );

    const renderCard = (title, icon, body) => (
        <div className="result-card">
            <div className="card-heading">
                <span className="card-icon">{icon}</span>
                <h2>{title}</h2>
            </div>
            {body}
        </div>
    );

    const structuredJson = {
        product_info: productInfo,
        analysis: aiAnalysis,
        recommendations,
        ocr: {
            text: ocrText,
            confidence: ocrConfidence,
        },
    };

    return (
        <div className="container">
            <style>{`
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 24px;
                    color: #1f2937;
                }
                .page-title {
                    margin-bottom: 20px;
                    font-size: 2rem;
                    font-weight: 800;
                    color: #0f172a;
                }
                .results-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                    gap: 16px;
                    align-items: start;
                }
                .result-card {
                    background: #ffffff;
                    border: 1px solid #e5e7eb;
                    border-radius: 18px;
                    padding: 18px;
                    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
                }
                .card-heading {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin-bottom: 14px;
                }
                .card-heading h2 {
                    margin: 0;
                    font-size: 1.05rem;
                }
                .card-icon {
                    font-size: 1.2rem;
                }
                .field-row {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    gap: 12px;
                    padding: 8px 0;
                    border-bottom: 1px solid #f1f5f9;
                }
                .ocr-preview {
                    margin-bottom: 12px;
                    padding: 10px;
                    border-radius: 12px;
                    background: #f8fafc;
                    border: 1px solid #e2e8f0;
                }
                .ocr-text-block {
                    margin: 8px 0 0;
                    white-space: pre-wrap;
                    font-family: inherit;
                    font-size: 0.9rem;
                    line-height: 1.5;
                    color: #0f172a;
                }
                .field-row:last-child {
                    border-bottom: none;
                }
                .field-label {
                    font-weight: 700;
                    color: #475569;
                }
                .badge {
                    display: inline-flex;
                    align-items: center;
                    padding: 4px 10px;
                    border-radius: 999px;
                    font-size: 0.82rem;
                    font-weight: 700;
                    white-space: normal;
                    text-align: center;
                }
                .badge.success {
                    background: #e8f8ee;
                    color: #157347;
                }
                .badge.warning {
                    background: #fff7dd;
                    color: #9a6a00;
                }
                .badge.danger {
                    background: #fee7e7;
                    color: #9d1c1c;
                }
                .badge.neutral {
                    background: #eff6ff;
                    color: #1d4ed8;
                }
                .developer-panel {
                    margin-top: 16px;
                }
                details {
                    border: 1px solid #dbeafe;
                    border-radius: 14px;
                    overflow: hidden;
                    background: #f8fbff;
                }
                details summary {
                    cursor: pointer;
                    padding: 12px 14px;
                    font-weight: 800;
                    color: #1e3a8a;
                }
                .developer-body {
                    padding: 0 14px 14px;
                }
                .json-block {
                    background: #0f172a;
                    color: #e2e8f0;
                    border-radius: 12px;
                    padding: 12px;
                    overflow: auto;
                    font-size: 0.85rem;
                }
                @media (max-width: 640px) {
                    .container { padding: 14px; }
                    .field-row { flex-direction: column; align-items: flex-start; }
                }
            `}</style>

            <h1 className="page-title">ShelfSense AI</h1>

            <UploadForm setResult={setResult} />

            {result && (
                <div className="results-grid">
                    {renderCard(
                        "� OCR Result",
                        "🔎",
                        <>
                            <h4>Cleaned OCR</h4>
                            <pre className="ocr-text-block">
                                {ocrText.join("\n") || "—"}
                            </pre>
                            <h4 style={{ marginTop: "18px" }}>
                                Reconstructed Text
                            </h4>

                            <pre className="ocr-text-block">
                                {reconstructedText || "—"}
                            </pre>
                            <h4>Raw OCR</h4>
                            <pre className="ocr-text-block">
                                {rawText.join("\n") || "—"}
                            </pre>
                        </>
                    )}

                    {renderCard(
                        "📦 Product Information",
                        "📦",
                        <>
                            {renderField("Product Name", productInfo.product_name || aiAnalysis.product_name, badgeTone(productInfo.product_name || aiAnalysis.product_name))}
                            {renderField("Manufacturer", productInfo.manufacturer || aiAnalysis.manufacturer, badgeTone(productInfo.manufacturer || aiAnalysis.manufacturer))}
                            {renderField("Category", productInfo.category || aiAnalysis.category, badgeTone(productInfo.category || aiAnalysis.category))}
                            {renderField("Ingredients", (productInfo.ingredients || aiAnalysis.ingredients || []).join(", ") || "—", badgeTone((productInfo.ingredients || aiAnalysis.ingredients || []).length ? "success" : ""))}
                            {renderField("Batch Number", batchNumber, badgeTone(batchNumber))}
                            {renderField("Manufacturing Date", manufacturingDate || productInfo?.confidence, badgeTone(manufacturingDate))}
                            {renderField("Expiry Date", productInfo.expiry_date || aiAnalysis.expiry_date, badgeTone(productInfo.expiry_date || aiAnalysis.expiry_date))}
                            {renderField("MRP", mrp, badgeTone(mrp))}
                        </>
                    )}

                    {renderCard(
                        "🧠 Product Intelligence",
                        "🧠",
                        <>
                            {renderField("OCR Quality", averageConfidence >= 80 ? "Good" : averageConfidence >= 60 ? "Moderate" : "Low", badgeTone(averageConfidence >= 80 ? "success" : averageConfidence >= 60 ? "warning" : "danger"))}
                            {renderField("Data Completeness", `${completenessScore}%`, badgeTone(completenessScore >= 80 ? "success" : completenessScore >= 50 ? "warning" : "danger"))}
                            {renderField("Expiry Status", recommendations.inventory_status || (productInfo.expiry_date ? "Valid" : "Unknown"), badgeTone(recommendations.inventory_status || (productInfo.expiry_date ? "success" : "warning")))}
                            {renderField("Manufacturer Verified", productInfo.manufacturer ? "Verified" : "Pending", badgeTone(productInfo.manufacturer ? "success" : "warning"))}
                            {renderField("Product Identified", productInfo.product_name ? "Identified" : "Not Identified", badgeTone(productInfo.product_name ? "success" : "danger"))}
                        </>
                    )}

                    {renderCard(
                        "🤖 AI Analysis",
                        "🤖",
                        <>
                            {renderField("Product Purpose", aiAnalysis.product_name ? "Label parsing and assessment" : "—", badgeTone("success"))}
                            {renderField("Product Category", aiAnalysis.category || productInfo.category || "—", badgeTone(aiAnalysis.category || productInfo.category))}
                            {renderField("Primary Use", aiAnalysis.category === "medicine" ? "Therapeutic" : aiAnalysis.category === "food" ? "Consumption" : "General Use", badgeTone("success"))}
                            {renderField("Confidence", aiAnalysis.confidence || productInfo.confidence || "—", badgeTone(aiAnalysis.confidence || productInfo.confidence))}
                            {renderField("Safety Notes", (aiAnalysis.warnings || productInfo.warnings || []).join(", ") || "No major warnings", badgeTone((aiAnalysis.warnings || productInfo.warnings || []).length ? "warning" : "success"))}
                        </>
                    )}

                    {renderCard(
                        "📌 Recommendation",
                        "📌",
                        <>
                            {renderField("Decision", recommendations.recommendation || "No recommendation", badgeTone(recommendations.risk_level || "warning"))}
                            {renderField("Risk Level", recommendations.risk_level || "Unknown", badgeTone(recommendations.risk_level || "warning"))}
                            {renderField("Summary", recommendations.recommendation || "—", badgeTone("success"))}
                            {renderField("Warnings", (productInfo.warnings || aiAnalysis.warnings || []).join(", ") || "None", badgeTone((productInfo.warnings || aiAnalysis.warnings || []).length ? "warning" : "success"))}
                            {renderField("Follow-up Actions", recommendations.priority === "High" ? "Review immediately" : recommendations.priority === "Medium" ? "Monitor closely" : "Continue standard checks", badgeTone(recommendations.priority === "High" ? "danger" : recommendations.priority === "Medium" ? "warning" : "success"))}
                        </>
                    )}
                </div>
            )}

            {result && (
                <div className="developer-panel">
                    <details>
                        <summary>🛠️ Developer Section</summary>
                        <div className="developer-body">
                            <div className="result-card">
                                <div className="card-heading">
                                    <span className="card-icon">📄</span>
                                    <h2>Structured Product JSON</h2>
                                </div>
                                <pre className="json-block">{JSON.stringify(structuredJson, null, 2)}</pre>
                            </div>

                            <div className="result-card" style={{ marginTop: 14 }}>
                                <div className="card-heading">
                                    <span className="card-icon">🔎</span>
                                    <h2>OCR Text</h2>
                                </div>
                                <pre className="json-block">{JSON.stringify(ocrText, null, 2)}</pre>
                            </div>
                        </div>
                    </details>
                </div>
            )}
        </div>
    );
}

export default App;