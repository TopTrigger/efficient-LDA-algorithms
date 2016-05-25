
# coding: utf-8

# In[1]:

import numpy as np
import scipy.io as sio
import sys
import logging
import time

rng = np.random.RandomState(seed=1234)


# In[2]:

def loglikelihood(doc_topic_matrix, topic_word_matrix, doc_word_matrix, alpha, beta):
    word_list, doc_list = doc_word_matrix.nonzero()
    theta = np.asarray(doc_topic_matrix, dtype=np.float)
    phi = np.asarray(topic_word_matrix.T, dtype=np.float)
    theta = theta + alpha
    theta_norm = np.sum(theta, axis=1)[:, np.newaxis]
    theta = theta / theta_norm
    phi = phi + beta
    phi_norm = np.sum(phi, axis=1)[:, np.newaxis]
    phi = phi / phi_norm

    ll = []
    for word_indx, doc_indx in zip(word_list, doc_list):
        product = np.dot(theta[doc_indx, :], phi[:, word_indx])
        ll.append(product)

    logll = np.sum(np.log(ll))

    return logll, theta, phi


# In[3]:

def baseline_lda(doc_word_matrix, K, iters, alpha, beta):
    num_voc, num_doc = doc_word_matrix.shape
    doc_topic_matrix = np.zeros([num_doc, K], dtype=np.int)
    topic_word_matrix = np.zeros([num_voc, K], dtype=np.int)
    topic_sum = np.zeros(K)

    word_list, doc_list = doc_word_matrix.nonzero()
    doc_word_assigned_topic = np.zeros(doc_word_matrix.shape, dtype=np.int)
    for word_indx, doc_indx in zip(word_list, doc_list):
        assigned_topic = rng.randint(K)
        doc_word_assigned_topic[word_indx, doc_indx] = assigned_topic
        doc_topic_matrix[doc_indx, assigned_topic] += 1
        topic_word_matrix[word_indx, assigned_topic] += 1
        topic_sum[assigned_topic] += 1

    st = time.time()
    logger.info('start training with baseline_lda k=%d max_iter=%d' % (K, iters))

    for i in range(iters):
        for word_indx, doc_indx in zip(word_list, doc_list):
            assigned_topic = doc_word_assigned_topic[word_indx, doc_indx]

            doc_topic_matrix[doc_indx, assigned_topic] -= 1
            topic_word_matrix[word_indx, assigned_topic] -= 1
            topic_sum[assigned_topic] -= 1

            conditional_probability = (topic_word_matrix[word_indx] + beta) / (topic_sum + num_voc * beta) * (doc_topic_matrix[doc_indx] + alpha)
            conditional_probability = conditional_probability / sum(conditional_probability)
            new_assigned_topic = int(rng.choice(K, 1, p=conditional_probability))

            doc_word_assigned_topic[word_indx, doc_indx] = new_assigned_topic
            doc_topic_matrix[doc_indx, new_assigned_topic] += 1
            topic_word_matrix[word_indx, new_assigned_topic] += 1
            topic_sum[new_assigned_topic] += 1

        iter_time = time.time() - st
        st = time.time()

        ll, theta, phi = loglikelihood(doc_topic_matrix, topic_word_matrix, doc_word_matrix, alpha, beta)
        ll_time = time.time() - st
        st = time.time()
        logger.info('iter %d sampling_time %f loglikelihood_time %f ll %f' % (i, iter_time, ll_time, ll))


# In[4]:

def sparse_lda(doc_word_matrix, K, iters, alpha, beta):
    num_voc, num_doc = doc_word_matrix.shape
    doc_topic_matrix = np.zeros((num_doc, K), dtype=np.int)
    topic_word_matrix = np.zeros((num_voc, K), dtype=np.int)
    topic_sum = np.zeros(K)
    Vbeta = num_voc * beta

    # random word topic assignment initialization
    word_list, doc_list = doc_word_matrix.nonzero()
    doc_word_assigned_topic = np.zeros(doc_word_matrix.shape, dtype=np.int)
    for word_indx, doc_indx in zip(word_list, doc_list):
        assigned_topic = rng.randint(K)
        doc_word_assigned_topic[word_indx, doc_indx] = assigned_topic
        doc_topic_matrix[doc_indx, assigned_topic] += 1
        topic_word_matrix[word_indx, assigned_topic] += 1
        topic_sum[assigned_topic] += 1


    # cache variable computed
    ssum = alpha * beta * np.sum(1/(topic_sum + Vbeta))
    q1 = alpha  / (topic_sum + Vbeta)

    st = time.time()
    logger.info('start training with sparse_lda k=%d max_iter=%d' % (K, iters))

    for i in range(iters):
        for doc_indx in xrange(num_doc):
            temp = doc_topic_matrix[doc_indx] / (topic_sum + Vbeta)
            q1 += temp
            rsum = beta * temp.sum()

            current_doc = doc_word_matrix.getcol(doc_indx)
            has_word_indices = current_doc.nonzero()[0]

            for word_indx in iter(has_word_indices):
                assigned_topic = doc_word_assigned_topic[word_indx, doc_indx]
                # remove chosen word-topic pair
                doc_topic_matrix[doc_indx, assigned_topic] -= 1
                topic_word_matrix[word_indx, assigned_topic] -= 1
                topic_sum[assigned_topic] -= 1

                # update the bucket sums
                denominator = topic_sum[assigned_topic] + Vbeta
                nt_d = doc_topic_matrix[doc_indx, assigned_topic]
                ssum = ssum + alpha * beta * (1 / denominator - 1 / (denominator - 1))
                rsum = rsum - (nt_d + 1) * beta / (denominator + 1) + (nt_d * beta) / denominator
                q1[assigned_topic] = (alpha + nt_d) / denominator
                p = topic_word_matrix[word_indx] * q1
                qsum = p.sum()

                total_sum = ssum + rsum + qsum
                U = rng.rand() * total_sum
                tmp = U

                if U < ssum:
                    for t in range(K):
                        U -= 1 / (topic_sum[t] + Vbeta)
                        if U <= 0:
                            new_assigned_topic = t
                            break

                elif U < (ssum + rsum):
                    U -= ssum
                    U /= beta
                    current_doc_topic = doc_topic_matrix[doc_indx]
                    for topic_indx in range(K):
                        U -= current_doc_topic[topic_indx] / (topic_sum[topic_indx] + Vbeta)
                        if U <= 0:
                            new_assigned_topic = topic_indx
                            break

                else:
                    U -= (ssum + rsum)
                    for topic_indx in range(K):
                        U -= p[topic_indx]
                        if U <= 0:
                            new_assigned_topic = topic_indx
                            break

                nt_d = doc_topic_matrix[doc_indx, new_assigned_topic]
                ssum = ssum + alpha * beta * (1 / (denominator + 1) - 1 / denominator)
                rsum = rsum - nt_d * beta / denominator + (nt_d + 1) * beta / (denominator + 1)
                q1[new_assigned_topic] = (alpha + nt_d + 1) / (denominator + 1)

                doc_word_assigned_topic[word_indx, doc_indx] = new_assigned_topic
                doc_topic_matrix[doc_indx, new_assigned_topic] += 1
                topic_word_matrix[word_indx, new_assigned_topic] += 1
                topic_sum[new_assigned_topic] += 1

            q1 -= doc_topic_matrix[doc_indx] / (topic_sum + Vbeta)


        iter_time = time.time() - st
        st = time.time()

        ll, theta, phi = loglikelihood(doc_topic_matrix, topic_word_matrix, doc_word_matrix, alpha, beta)
        ll_time = time.time() - st
        st = time.time()
        logger.info('iter %d sampling_time %f loglikelihood_time %f ll %f' % (i, iter_time, ll_time, ll))


# In[5]:

class AliasTable:
    def __init__(self, bins):
        self.bins = bins
        self.sample_times = -1
        self.table = np.zeros((2, bins))
        self.prob_sum = 0
        self.prob = np.zeros((bins))

    def construct(self, prob):
        # unnormalized probability
        self.prob = prob
        self.prob_sum = self.prob.sum()
        p = self.prob * self.bins / self.prob_sum

        small = [i for i in range(self.bins) if p[i] < 1]
        large = [i for i in range(self.bins) if p[i] >=1]

        while(not (small == [] or large == [])):
            l = small.pop()
            g = large.pop()
            self.table[0, l] = p[l]
            self.table[1, l] = g
            p[g] = p[g] - (1 - p[l])
            if p[g] < 1:
                small.append(g)
            else:
                large.append(g)

        while(large != []):
            g = large.pop()
            self.table[0, g] = 1

        while(small != []):
            l = small.pop()
            self.table[0, l] = 1

        self.sample_times = 0

    def sample(self, randint, rand):
        if self.table[0, randint] > rand:
            return randint
        else:
            return int(self.table[1, randint])


# In[6]:

def alias_lda(doc_word_matrix, K, iters, alpha, beta):
    num_voc, num_doc = doc_word_matrix.shape
    doc_topic_matrix = np.zeros((num_doc, K), dtype=np.int)
    topic_word_matrix = np.zeros((num_voc, K), dtype=np.int)
    topic_sum = np.zeros(K)
    Vbeta = num_voc * beta

    # random word topic assignment initialization
    word_list, doc_list = doc_word_matrix.nonzero()
    doc_word_assigned_topic = np.zeros(doc_word_matrix.shape, dtype=np.int)
    for word_indx, doc_indx in zip(word_list, doc_list):
        assigned_topic = rng.randint(K)
        doc_word_assigned_topic[word_indx, doc_indx] = assigned_topic
        doc_topic_matrix[doc_indx, assigned_topic] += 1
        topic_word_matrix[word_indx, assigned_topic] += 1
        topic_sum[assigned_topic] += 1

    # initialize alias tables for each word against topics
    AT = []
    for word_indx in range(num_voc):
        t = AliasTable(K)
        prob = alpha * (topic_word_matrix[word_indx] + beta) / (topic_sum + Vbeta)
        t.construct(prob)
        AT.append(t)


    st = time.time()
    logger.info('start training with alias_lda k=%d max_iter=%d' % (K, iters))

    for i in range(iters):
        for doc_indx in xrange(num_doc):
            current_doc = doc_word_matrix.getcol(doc_indx)
            has_word_indices = current_doc.nonzero()[0]

            for word_indx in iter(has_word_indices):
                assigned_topic = doc_word_assigned_topic[word_indx, doc_indx]
                # remove chosen word-topic pair
                doc_topic_matrix[doc_indx, assigned_topic] -= 1
                topic_word_matrix[word_indx, assigned_topic] -= 1
                topic_sum[assigned_topic] -= 1

                # compute p, doc-topic prob
                p_dw = doc_topic_matrix[doc_indx] * (topic_word_matrix[word_indx] + beta) / (topic_sum + Vbeta)
                p_sum = p_dw.sum()
                w_table = AT[word_indx]
                q_sum = w_table.prob_sum

                # sample new topic
                for ii in range(2):
                    if rng.rand() < p_sum / (p_sum + q_sum):
                        u = rng.rand() * p_sum
                        for new_topic in range(K):
                            u -= p_dw[new_topic]
                            if u < 0:
                                break

                    else:
                        w_table.sample_times += 1
                        if w_table.sample_times > K:
                            # if alias table sample more than K times, update
                            w_table = AliasTable(K)
                            prob = alpha * (topic_word_matrix[word_indx] + beta) / (topic_sum + Vbeta)
                            w_table.construct(prob)
                            AT[word_indx] = w_table

                        new_topic = w_table.sample(rng.randint(K), rng.rand())

                    if new_topic != assigned_topic:
                        q_w = w_table.prob

                        temp_old = (topic_word_matrix[word_indx][assigned_topic] + beta) / (topic_sum[assigned_topic] + Vbeta)
                        temp_new = (topic_word_matrix[word_indx][new_topic] + beta) / (topic_sum[new_topic] + Vbeta)

                        acceptance = (doc_topic_matrix[doc_indx][new_topic] + alpha) / (doc_topic_matrix[doc_indx][assigned_topic] + alpha)                                     * temp_new / temp_old                                     * (doc_topic_matrix[doc_indx][assigned_topic] * temp_old + q_w[assigned_topic])                                     / (doc_topic_matrix[doc_indx][new_topic] * temp_new + q_w[new_topic])

                        if rng.rand() < acceptance:
                            assigned_topic = new_topic

                doc_word_assigned_topic[word_indx, doc_indx] = assigned_topic
                doc_topic_matrix[doc_indx, assigned_topic] += 1
                topic_word_matrix[word_indx, assigned_topic] += 1
                topic_sum[assigned_topic] += 1


        iter_time = time.time() - st
        st = time.time()

        ll, theta, phi = loglikelihood(doc_topic_matrix, topic_word_matrix, doc_word_matrix, alpha, beta)
        ll_time = time.time() - st
        st = time.time()
        logger.info('iter %d sampling_time %f loglikelihood_time %f ll %f' % (i, iter_time, ll_time, ll))


# In[7]:

def main(argv):
    if len(argv) != 5:
        print "Usage: python efficient_lda.py <lda_type> <K> <max_iter> <log_name>"
    else:
        lda_type = str(argv[1])
        K = int(argv[2])
        max_iter = int(argv[3])
        log_name = str(argv[4])

        alpha = 50.0 / K
        beta = 0.1

        infos = sio.loadmat('nips12raw_str602.mat')
        doc_word_matrix = infos['counts']

        global logger
        logger = logging.getLogger()


        logging.basicConfig(filename=log_name, level=logging.INFO, format='%(asctime)s - %(message)s')

        if lda_type == 'baseline':
            baseline_lda(doc_word_matrix, K, max_iter, alpha, beta)
        elif lda_type == 'sparse':
            sparse_lda(doc_word_matrix, K, max_iter, alpha, beta)
        elif lda_type == 'alias':
            alias_lda(doc_word_matrix, K, max_iter, alpha, beta)
        else:
            print "Not implemented type!"


# In[8]:

if __name__ == "__main__":
    main(sys.argv)

