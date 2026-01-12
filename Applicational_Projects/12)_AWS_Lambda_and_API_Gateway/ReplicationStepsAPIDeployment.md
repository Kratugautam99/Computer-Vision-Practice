## AWS Lambda, S3, and API Gateway Setup Guide
### 1. **Login to AWS Console**
- Go to [https://aws.amazon.com](https://aws.amazon.com)
- Sign in with your AWS credentials.
- Navigate to the **AWS Management Console**.

---

### 2. **Create an S3 Bucket and Upload `S3Bucket.zip`**

#### Step 2.1: Create an S3 Bucket
- Go to **Services** > **Storage** > **S3**.
- Click **Create bucket**.
- Provide a unique bucket name (e.g., `lambda-layer-bucket-123`).
- Select a region.
- Click **Create bucket**.

#### Step 2.2: Upload `S3Bucket.zip`
- Navigate to your bucket.
- Click **Upload**.
- Select `S3Bucket.zip` from your local directory.
- Click **Upload**.

#### Step 2.3: Copy the S3 Object URL
- After upload, copy the **Object URL** (e.g., `https://s3.amazonaws.com/your-bucket/S3Bucket.zip`).
- Note: You may need to make the object public or use pre-signed URLs depending on access requirements.

---

### 3. **Create a Lambda Layer**

#### Step 3.1: Navigate to Lambda Layers
- Go to **Services** > **Lambda** > **Layers**.
- Click **Create layer**.

#### Step 3.2: Configure the Layer
- **Name**: `opencv-layer`
- **Compatible runtimes**: Select the runtime matching your Lambda function (e.g., Python 3.9).
- **Upload from**: Select **Upload from** and upload `S3Bucket.zip` from S3.
- **Layer version**: Provide a description.
- Click **Create**.

#### Step 3.3: Copy the Layer ARN
- After creation, copy the **Layer ARN** (e.g., `arn:aws:lambda:eu-north-1:123456789012:layer:opencv-layer:1`).

---

### 4. **Create a Lambda Function**

#### Step 4.1: Navigate to Lambda
- Go to **Services** > **Lambda** > **Functions**.
- Click **Create function**.

#### Step 4.2: Configure the Function
- **Function name**: `image-grayscale-lambda`
- **Runtime**: Python 3.9 (or compatible)
- **Architecture**: x86_64 or arm64 (as needed)
- Click **Create function**.

#### Step 4.3: Upload `LambdaFunctionCode.py`
- In the **Function code** section, click **Upload from**.
- Select **File** and upload `LambdaFunctionCode.py` from your local directory.
- Ensure the **Handler** is set to `LambdaFunctionCode.lambda_handler`.

#### Step 4.4: Add the Lambda Layer
- Scroll to the **Layers** section.
- Click **Add a layer**.
- Select **Custom layer**.
- Paste the ARN of the layer you created (`opencv-layer`).
- Click **Add**.

#### Step 4.5: Save Changes
- Click **Save** at the top.

---

### 5. **Test the Lambda Function**

#### Step 5.1: Create a Test Event
- In the Lambda console, click **Test**.
- Create a test event with a base64-encoded image string (e.g., from `TestIMG.png`).
- Use the default test event template and modify the `body` field to include a valid base64 string.

#### Step 5.2: Run the Test
- Click **Test**.
- Verify the function returns a valid response with a base64-encoded grayscale image.

---

### 6. **Create an API Gateway (REST API)**

#### Step 6.1: Navigate to API Gateway
- Go to **Services** > **API Gateway** > **REST APIs**.
- Click **Create API** > **REST API**.
- Choose **New API**.
- Provide a name (e.g., `ImageConverterAPI`).
- Click **Create API**.

#### Step 6.2: Create a Resource and Method
- Click **Create Resource**.
- Name it `/convert`.
- Click **Create Method**.
- Select **POST**.
- Choose **Lambda Function** as the integration type.
- Enter the Lambda function ARN (e.g., `arn:aws:lambda:eu-north-1:123456789012:function:image-grayscale-lambda`).
- Click **Save**.

#### Step 6.3: Set Binary Media Types
- In the API Gateway console, go to **Settings**.
- Under **Binary Media Types**, add `*/*`.
- Click **Save**.

---

### 7. **Deploy the API**

#### Step 7.1: Deploy the API
- Go to **Actions** > **Deploy API**.
- Select **[New Stage]**.
- Name the stage (e.g., `prod`).
- Click **Deploy**.

#### Step 7.2: Copy the Invoke URL
- After deployment, copy the **Invoke URL** (e.g., `https://abc123.execute-api.eu-north-1.amazonaws.com/prod/convert`).

---

### 8. **Test the API Using `main.py`**

#### Step 8.1: Run `main.py`
- Ensure `main.py` is in the same directory as `TestIMG.png`.
- Run the script to test the full pipeline.
- The output will be saved as `OutputIMG.png` in the same directory.

---

### 9. **Verify Output**
- Open `OutputIMG.png` to confirm it is a grayscale version of `TestIMG.png`.

---

### 10. **Cleanup (Optional)**
- Delete the S3 bucket if no longer needed.
- Delete the Lambda function and API Gateway if not required.
