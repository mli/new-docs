stage("Document") {
  node {
    ws('workspace/mxnet-new-docs') {
      checkout scm
      sh "build/build.sh"
      sh """#!/bin/bash
      set -ex
      if [[ ${env.BRANCH_NAME} == master ]]; then
          conda activate mxnet-docs
          aws s3 sync --delete build/_build/html/ s3://beta.mxnet.io/ --acl public-read
      fi
      """
    }
  }
}
