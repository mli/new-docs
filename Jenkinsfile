stage('Python Docs') {
  node('linux-gpu') {
    ws('workspace/mxnet-new-docs') {
      checkout scm
      sh "conda env update -f environment.yml"
      sh "python/build_doc.sh build/html"
      sh "Rsite/build_doc.sh build/html/r"
      if (env.BRANCH_NAME == 'master') {
        sh "build/upload_doc.sh build/html s3://beta.mxnet.io/"
      }
    }
  }
}
