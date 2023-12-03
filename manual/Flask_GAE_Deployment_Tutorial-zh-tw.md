🇬🇧 [English](./Flask_GAE_Deployment_Tutorial.md) | 🇹🇼 [繁體中文](./Flask_GAE_Deployment_Tutorial-zh.tw.md)

# 將 Flask API 部署到 Google App Engine

這份指南提供了一個逐步過程，說明如何將基於 Flask 的 API 部署到 Google App Engine (GAE)，讓您能夠高效地托管和管理您的應用程式。

## 先決條件

- 在本地環境開發的 Flask API。
- 安裝 Python 3.6 或更高版本。
- 在您的機器上安裝 Google Cloud SDK。
- 一個 Google Cloud 帳戶。

## 步驟 1：準備您的 Flask 應用程式

確保您的 Flask 應用程式在本地環境中運行正常。應用程式應該處於可部署的完整狀態。

## 步驟 2：創建 `app.yaml` 配置文件

在您的 Flask 應用程式的根目錄中，創建一個名為 `app.yaml` 的文件。這個文件包含 GAE 的配置設置。對於基本的 Flask 應用程式，內容應該是：

```yaml
runtime: python39
entrypoint: gunicorn -b :$PORT main:app
```

將 `main:app` 替換為您 Flask 應用的適當模組和應用名稱。

## 步驟 3：配置 `requirements.txt`

確保您的 `requirements.txt` 文件列出了所有必要的依賴項。它至少應該包括 Flask 和 Gunicorn：

```
Flask==X.X.X
gunicorn==X.X.X
# 其他依賴...
```

## 步驟 4：安裝 Google Cloud SDK

如果尚未安裝，請下載並安裝 Google Cloud SDK。這個工具對於在 GAE 上部署和管理您的應用程式至關重要。

## 步驟 5：初始化您的 GAE 項目

執行以下命令來初始化您的 GAE 配置：

```bash
gcloud init
```

按照提示選擇或創建一個新的 Google Cloud 項目並設置預設區域。

## 步驟 6：部署您的應用程式

導航至您的 Flask 應用程式目錄並執行：

```bash
gcloud app deploy
```

這個命令將上傳並部署您的應用程式到 GAE。部署成功後，您將收到一個 URL 以訪問您的應用。

## 步驟 7：訪問您的應用程式

在瀏覽器中打開提供的 URL。您的 Flask 應用程式現在應該在線上運行。

## 額外注意事項

- 確保 App Engine 服務在您的 Google Cloud 項目中已啟用。
- 定期監控您的應用程式以確保其按預期運行。
- 考慮您的應用程式的安全性和資源使用情況，尤其是如果預計會有較高的流量。

部署到 GAE 允許您專注於開發，同時由 Google 處理伺服器擴展和維護等基礎設施管理任務。
這份指南涵蓋了將 Flask API 部署到 Google App Engine 的基本步驟，適合對 Flask 和 Google Cloud 服務有基本了解的開發人員。