import { useState } from "react";
import api from "../services/api";

function UploadForm({ setResult }) {

    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleFileChange = (e) => {

        const selectedFile = e.target.files[0];

        if (!selectedFile) return;

        setFile(selectedFile);
        setPreview(URL.createObjectURL(selectedFile));
    };

    const uploadImage = async () => {

        if (!file) {
            alert("Please choose an image.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        try {

            setLoading(true);

            const response = await api.post(
                "/upload",
                formData,
                {
                    headers: {
                        "Content-Type": "multipart/form-data"
                    }
                }
            );

            setResult(response.data);

        } catch (error) {

            console.error(error);

            alert("Upload failed.");

        } finally {

            setLoading(false);

        }

    };

    return (

        <div>

            <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
            />

            <br /><br />

            <button onClick={uploadImage}>
                Analyze Product
            </button>

            {loading && <p>Analyzing...</p>}

            {preview && (

                <img
                    src={preview}
                    alt="Preview"
                    style={{
                        width: "300px",
                        marginTop: "20px",
                        borderRadius: "10px"
                    }}
                />

            )}

        </div>

    );

}

export default UploadForm;