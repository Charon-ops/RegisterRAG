document.addEventListener("DOMContentLoaded", function () {
  const uploadTab = document.getElementById("upload-tab");
  const responseTab = document.getElementById("response-tab");
  const subTitle = document.getElementById("sub-title");

  const uploadContainer = document.getElementById("upload");

  // Default select upload tab
  if (uploadTab) {
    uploadTab.style.backgroundColor = "lightblue";
  }
  if (subTitle) {
    subTitle.textContent = "Upload";
  }

  // Switch to upload tab
  uploadTab?.addEventListener("click", function () {
    uploadTab.style.backgroundColor = "lightblue";
    if (responseTab) {
      responseTab.style.backgroundColor = "transparent";
    }
    if (subTitle) {
      subTitle.textContent = "Upload";
    }
    if(uploadContainer) {
      uploadContainer.style.visibility = "visible"
    }
  });

  // Switch to response tab
  responseTab?.addEventListener("click", function () {
    responseTab.style.backgroundColor = "lightblue";
    if (uploadTab) {
      uploadTab.style.backgroundColor = "transparent";
    }
    if (subTitle) {
      subTitle.textContent = "Response";
    }
    if (uploadContainer) {
      uploadContainer.style.visibility = "hidden";
    }
  });
});
