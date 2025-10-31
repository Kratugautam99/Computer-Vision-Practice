# 🔑 Steps to Get AWS Rekognition Access Keys

1. **Sign in to AWS Console**  
   - Go to [https://console.aws.amazon.com](https://console.aws.amazon.com) and log in with your AWS account.

2. **Open the IAM (Identity and Access Management) service**  
   - In the search bar, type **IAM** and select it.

3. **Create or choose a user**  
   - If you don’t already have a programmatic user, click **Users → Add users**.  
   - Enter a username (e.g., `rekognition-user`).  
   - Under **Access type**, select **Programmatic access**.

4. **Attach permissions**  
   - Choose **Attach existing policies directly**.  
   - Select **AmazonRekognitionFullAccess** (or a custom policy with the required permissions).  
   - Click **Next** until you can **Create user**.

5. **Copy the credentials**  
   - After creation, you’ll see an **Access Key ID** and a **Secret Access Key**.  
   - Copy them or download the `.csv` file.  
   - ⚠️ The **Secret Access Key is shown only once** — make sure to save it securely.
