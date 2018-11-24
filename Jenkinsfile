stage("Document") {
  node {
    ws('workspace/mxnet-new-docs') {
      checkout scm
      sh "build/build.sh"
    }
  }
}
