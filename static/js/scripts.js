// static/js/scripts.js

document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const loadingSpinner = document.getElementById('loading-spinner');
    const resultsDiv = document.getElementById('processed-image-container');
    const processedImage = document.getElementById('processed-image');
    const processedImageLink = document.getElementById('processed-image-link');
    const modalImage = document.getElementById('modal-image');
    const alertPlaceholder = document.getElementById('alert-placeholder');
    const downloadImageButton = document.getElementById('download-image');
    const downloadCsvButton = document.getElementById('download-csv');
    const downloadPdfButton = document.getElementById('download-pdf');

    const volumeInfoDiv = document.getElementById('volume-info');
    const volumeTableBody = document.querySelector('#volume-table tbody');
    const totalVolumeCell = document.getElementById('total-volume-cell');

    // Function to show alerts
    function showAlert(message, type='success') {
        const wrapper = document.createElement('div');
        wrapper.innerHTML = `
            <div class="alert alert-${type} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
        alertPlaceholder.append(wrapper);
    }

    // Function to clear alerts
    function clearAlerts() {
        alertPlaceholder.innerHTML = '';
    }

    // Populate the volume table with segments data
    function populateVolumeTable(data) {
        // Clear existing rows
        volumeTableBody.innerHTML = '';

        // Add rows for each segment
        data.segments.forEach(seg => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${seg.segment_no}</td>
                <td>${seg.volume.toFixed(2)}</td>
                <td>${seg.width.toFixed(2)}</td>
                <td>${seg.length.toFixed(2)}</td>
                <td>${seg.height.toFixed(2)}</td>
            `;
            volumeTableBody.appendChild(tr);
        });

        // Set total volume
        totalVolumeCell.textContent = `Total Volume: ${data.total_volume.toFixed(2)} cc`;
    }

    // Handle form submission
    uploadForm.addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent default form submission

        clearAlerts(); // Clear any existing alerts

        // Get the selected file
        const fileInput = document.getElementById('camera');
        const file = fileInput.files[0];

        if (!file) {
            showAlert('Please select or capture an image before submitting.', 'warning');
            return;
        }

        // Show loading spinner
        loadingSpinner.style.display = 'block';

        // Prepare form data
        const formData = new FormData();
        formData.append('image', file);

        // Send AJAX request to the server
        fetch('/process', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading spinner
            loadingSpinner.style.display = 'none';

            if (data.error) {
                showAlert(`Error: ${data.error}`, 'danger');
                return;
            }

            if (data.message) {
                showAlert(data.message, data.image ? 'success' : 'info');
            }

            if (data.segments && data.segments.length > 0 && data.image) {
                // Display processed image
                processedImage.src = `data:image/png;base64,${data.image}`;
                processedImageLink.href = `data:image/png;base64,${data.image}`;
                modalImage.src = `data:image/png;base64,${data.image}`;
                resultsDiv.style.display = 'block';

                // Populate the volume table
                populateVolumeTable(data);

                // Show the volume info div (table)
                volumeInfoDiv.style.display = 'block';

            } else {
                showAlert('No segments detected.', 'info');
            }
        })
        .catch(error => {
            // Hide loading spinner
            loadingSpinner.style.display = 'none';
            console.error('Error:', error);
            showAlert('An unexpected error occurred while processing the image.', 'danger');
        });
    });

    // Download the processed image
    downloadImageButton.addEventListener('click', function() {
        if (!processedImage.src) {
            showAlert('No processed image available to download.', 'warning');
            return;
        }

        const link = document.createElement('a');
        link.href = processedImage.src;
        link.download = 'processed_image.png';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    // Download volume data as CSV
    downloadCsvButton.addEventListener('click', function() {
        const rows = document.querySelectorAll('#volume-table tbody tr');
        if (rows.length === 0) {
            showAlert('No volume data available to download.', 'warning');
            return;
        }

        const headers = ['Segment No.', 'Volume (cc)', 'Width (cm)', 'Length (cm)', 'Height (cm)'];
        const csvRows = [headers.join(',')];

        rows.forEach(row => {
            const cells = row.querySelectorAll('td');
            const rowData = Array.from(cells).map(cell => cell.textContent);
            csvRows.push(rowData.join(','));
        });

        const totalText = totalVolumeCell.textContent.replace('Total Volume: ', '');
        csvRows.push(`Total Volume,,,,${totalText}`);

        const csvContent = csvRows.join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.href = url;
        link.download = 'volume_data.csv';
        link.style.display = 'none';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    // Download volume data as PDF
    downloadPdfButton.addEventListener('click', function() {
        const rows = document.querySelectorAll('#volume-table tbody tr');
        if (rows.length === 0) {
            showAlert('No volume data available to download.', 'warning');
            return;
        }

        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();

        // Add title
        doc.setFontSize(16);
        doc.text('Volume Calculations', 14, 22);

        doc.setFontSize(12);
        let y = 30;

        // Headers
        doc.text('Segment No. | Volume (cc) | Width (cm) | Length (cm) | Height (cm)', 14, y);
        y += 10;

        // Rows
        rows.forEach(row => {
            const cells = row.querySelectorAll('td');
            const rowData = Array.from(cells).map(cell => cell.textContent);
            doc.text(rowData.join(' | '), 14, y);
            y += 10;
        });

        // Total volume
        const totalText = totalVolumeCell.textContent;
        doc.text(totalText, 14, y);

        doc.save('volume_data.pdf');
    });
});
