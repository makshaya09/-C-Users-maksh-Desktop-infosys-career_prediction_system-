// Client-side interactions and validation for AI Career Prediction System

document.addEventListener("DOMContentLoaded", function () {
  // 1. File Upload Drag-and-Drop Handling
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("resumeFile");
  const fileNameDisplay = document.getElementById("fileNameDisplay");

  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());

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

  function updateFileName(file) {
    if (fileNameDisplay && file) {
      const ext = file.name.split(".").pop().toLowerCase();
      if (!["pdf", "txt"].includes(ext)) {
        fileNameDisplay.innerHTML = `<span class="text-danger font-weight-bold">Unsupported file format (.${ext}). Please select a PDF or TXT file.</span>`;
        fileInput.value = "";
        return;
      }
      const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
      fileNameDisplay.innerHTML = `<strong>Selected file:</strong> ${file.name} (${sizeMb} MB)`;
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
