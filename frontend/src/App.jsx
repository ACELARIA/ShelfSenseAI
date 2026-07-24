import { useState } from "react";
import UploadForm from "./components/UploadForm";
import "./App.css";

function App() {

    const [result, setResult] = useState(null);

    return (

        <div className="container">

            <h1>ShelfSense AI</h1>

            <UploadForm setResult={setResult} />

            {result && (

                <>

                    <div className="card">

                        <h2>OCR Text</h2>

                        <pre>
{result.ocr.text.join("\n")}
                        </pre>

                    </div>

                    <div className="card">

                        <h2>AI Analysis</h2>

                        <pre>
{JSON.stringify(result.analysis, null, 2)}
                        </pre>

                    </div>

                    <div className="card">

                        <h2>Shelf Intelligence</h2>

                        <pre>
{JSON.stringify(result.insights, null, 2)}
                        </pre>

                    </div>

                </>

            )}

        </div>

    );

}

export default App;