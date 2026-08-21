// Client-side interactions and validation for AI Career Prediction System

document.addEventListener("DOMContentLoaded", function () {
  // 1. File Upload Drag-and-Drop Handling
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("resumeFile");
  const fileNameDisplay = document.getElementById("fileNameDisplay");
  const uploadForm = document.getElementById("uploadForm");

  if (dropzone && fileInput) {
    dropzone.addEventListener("click", function (e) {
      if (e.target !== fileInput) {
        fileInput.click();
      }
    });

    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
      dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        updateFileName(fileInput.files[0]);
      }
    });

    fileInput.addEventListener("change", () => {
      if (fileInput.files && fileInput.files.length > 0) {
        updateFileName(fileInput.files[0]);
      }
    });
  }

  if (uploadForm && fileInput) {
    uploadForm.addEventListener("submit", function (e) {
      if (!fileInput.files || fileInput.files.length === 0) {
        e.preventDefault();
        if (fileNameDisplay) {
          fileNameDisplay.innerHTML = `<span class="text-danger fw-bold">Please select a PDF or TXT resume file first.</span>`;
        }
        return;
      }
      const submitBtn = uploadForm.querySelector("button[type='submit']");
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Parsing Resume...`;
      }
    });
  }

  function updateFileName(file) {
    if (fileNameDisplay && file) {
      const ext = file.name.split(".").pop().toLowerCase();
      if (!["pdf", "txt"].includes(ext)) {
        fileNameDisplay.innerHTML = `<span class="text-danger fw-bold">Unsupported format (.${ext}). Please select a PDF or TXT file.</span>`;
        fileInput.value = "";
        return;
      }
      const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
      fileNameDisplay.innerHTML = `<span class="badge bg-success-subtle text-success border border-success-subtle p-2 px-3 fs-6"><i class="bi bi-file-earmark-check me-1"></i> ${file.name} (${sizeMb} MB)</span>`;
    }
  }

  // 2. Profile Form Validation
  const profileForm = document.getElementById("profileForm");
  if (profileForm) {
    profileForm.addEventListener("submit", function (e) {
      const nameInput = document.getElementById("name");
      const emailInput = document.getElementById("email");
      const educationInput = document.getElementById("education");
      const skillsInput = document.getElementById("skills");
      const experienceInput = document.getElementById("experience");

      let isValid = true;
      let errorMessages = [];

      // Validate Name
      if (nameInput && !nameInput.value.trim()) {
        isValid = false;
        errorMessages.push("Name cannot be empty.");
      }

      // Validate Email
      if (emailInput) {
        const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailPattern.test(emailInput.value.trim())) {
          isValid = false;
          errorMessages.push("Please enter a valid email address.");
        }
      }

      // Validate Education
      if (educationInput && !educationInput.value.trim()) {
        isValid = false;
        errorMessages.push("Education / Degree is required.");
      }

      // Validate Skills
      if (skillsInput && !skillsInput.value.trim()) {
        isValid = false;
        errorMessages.push("At least one skill is required.");
      }

      // Validate Experience
      if (experienceInput) {
        const expVal = parseFloat(experienceInput.value);
        if (isNaN(expVal)) {
          isValid = false;
          errorMessages.push("Years of experience must be a numeric value.");
        } else if (expVal < 0) {
          isValid = false;
          errorMessages.push("Years of experience cannot be negative.");
        }
      }

      if (!isValid) {
        e.preventDefault();
        const errorContainer = document.getElementById("clientErrorContainer");
        if (errorContainer) {
          errorContainer.innerHTML = `
            <div class="alert alert-danger alert-dismissible fade show" role="alert">
              <strong>Please correct the following errors:</strong>
              <ul class="mb-0 mt-1">
                ${errorMessages.map((msg) => `<li>${msg}</li>`).join("")}
              </ul>
              <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
          `;
          errorContainer.scrollIntoView({ behavior: "smooth" });
        }
      }
    });
  }
});
