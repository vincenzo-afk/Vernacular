/** Browser-only: hands a generated file to the user as a download. */
export function downloadFile(
  filename: string,
  data: string | Uint8Array,
  mime: string
): void {
  // Uint8Array -> Blob needs a plain ArrayBuffer-backed view.
  const part: BlobPart = typeof data === "string" ? data : new Uint8Array(data);
  const url = URL.createObjectURL(new Blob([part], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Revoke on the next tick so the download has started.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
